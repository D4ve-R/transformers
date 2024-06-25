import math
import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint as checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import logging

from .configuration_davit import DaViTConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DaViTConfig"


class DaViTSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DaViTPreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm is not None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size


class DaViTMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, hidden_features)),
                    ("act", act_layer()),
                    ("fc2", nn.Linear(hidden_features, out_features)),
                ]
            )
        )

    def forward(self, x, size):
        return self.net(x), size


class DaViTDepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class DaViTConvEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None, pre_norm=True):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = x.view(-1, H, W, x.size(-1)).permute(0, 3, 1, 2)

        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.size()
        x = x.view(B, -1, C)

        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class DaViTChannelAttention(nn.Module):
    def __init__(self, dim, groups=8, qkv_bias=True):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (float(N) ** -0.5)
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size

# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L150
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DaViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# https://github.com/huggingface/pytorch-image-models/blob/b28945ff056d454b174f0fb8682e362b87150141/timm/layers/weight_init.py
def trunc_normal_(tensor: torch.Tensor, mean=0.0, std=1.0, a=-2.0, b=2.0) -> torch.Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """

    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Cut & paste from PyTorch official master until it's in a few official releases - RW
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn(
                "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect.",
                stacklevel=2,
            )

        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class DaViTChannelBlock(nn.Module):
    def __init__(
        self,
        dim,
        groups,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
    ):
        super().__init__()

        drop_path = DaViTDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv1 = DaViTPreNorm(None, DaViTDepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = DaViTPreNorm(
            norm_layer(dim), DaViTChannelAttention(dim, groups=groups, qkv_bias=qkv_bias), drop_path
        )
        self.conv2 = DaViTPreNorm(None, DaViTDepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = DaViTPreNorm(
            norm_layer(dim), DaViTMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer), drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, batch_size: int, window_size: int, H: int, W: int):
    B = batch_size
    # this will cause onnx conversion failed for dynamic axis, because treated as constant
    # int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DaViTWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class DaViTSpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
    ):
        super().__init__()

        drop_path = DaViTDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv1 = DaViTPreNorm(None, DaViTDepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = DaViTPreNorm(
            norm_layer(dim), DaViTWindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias), drop_path
        )
        self.conv2 = DaViTPreNorm(None, DaViTDepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = DaViTPreNorm(
            norm_layer(dim), DaViTMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer), drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    """DaViT: Dual-Attention Transformer

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        patch_size (tuple(int)): Patch size of convolution in different stages. Default: (7, 2, 2, 2).
        patch_stride (tuple(int)): Patch stride of convolution in different stages. Default: (4, 2, 2, 2).
        patch_padding (tuple(int)): Patch padding of convolution in different stages. Default: (3, 0, 0, 0).
        patch_prenorm (tuple(bool)): If True, perform norm before convlution layer. Default: (True, False, False, False).
        embed_dims (tuple(int)): Patch embedding dimension in different stages. Default: (64, 128, 192, 256).
        num_heads (tuple(int)): Number of spatial attention heads in different stages. Default: (4, 8, 12, 16).
        num_groups (tuple(int)): Number of channel groups in different stages. Default: (4, 8, 12, 16).
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        enable_checkpoint (bool): If True, enable checkpointing. Default: False.
        conv_at_attn (bool): If True, performe depthwise convolution before attention layer. Default: True.
        conv_at_ffn (bool): If True, performe depthwise convolution before ffn layer. Default: True.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = DaViTConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i],
            )
            convs.append(conv_embed)

            block = DaViTSequential(
                *[
                    DaViTSequential(
                        OrderedDict(
                            [
                                (
                                    "spatial_block",
                                    DaViTSpatialBlock(
                                        embed_dims[i],
                                        num_heads[i],
                                        window_size,
                                        drop_path_rate=dpr[depth_offset + j * 2],
                                        qkv_bias=qkv_bias,
                                        mlp_ratio=mlp_ratio,
                                        conv_at_attn=conv_at_attn,
                                        conv_at_ffn=conv_at_ffn,
                                    ),
                                ),
                                (
                                    "channel_block",
                                    DaViTChannelBlock(
                                        embed_dims[i],
                                        num_groups[i],
                                        drop_path_rate=dpr[depth_offset + j * 2 + 1],
                                        qkv_bias=qkv_bias,
                                        mlp_ratio=mlp_ratio,
                                        conv_at_attn=conv_at_attn,
                                        conv_at_ffn=conv_at_ffn,
                                    ),
                                ),
                            ]
                        )
                    )
                    for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            for name, _ in m.named_parameters():
                if name in ["bias"]:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features_unpool(self, x):
        """
        forward until avg pooling
        Args:
            x (_type_): input image tensor
        """
        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            if self.enable_checkpoint:
                x, input_size = checkpoint.checkpoint(block, x, input_size)
            else:
                x, input_size = block(x, input_size)
        return x

    def forward_features(self, x):
        x = self.forward_features_unpool(x)

        # (batch_size, num_tokens, token_dim)
        x = self.avgpool(x.transpose(1, 2))
        # (batch_size, 1, num_tokens)
        x = torch.flatten(x, 1)
        x = self.norms(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            depths=config.depths,
            embed_dims=config.dim_embed,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            window_size=config.window_size,
        )

