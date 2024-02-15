
""" SincNet model """
from math import pi
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_sincnet import SincNetConfig
from ...modeling_utils import PreTrainedModel

# General docstring
_CONFIG_FOR_DOC = "SincNetConfig"

SINCNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all SincNet models at https://huggingface.co/models?filter=sincnet
]

# https://github.com/mravanelli/SincNet/blob/master/dnn_models.py
class SincNetFilterConvLayer(nn.Module):
    """Sinc fast convolution"""

    def __init__(self, out_channels: int, kernel_size: int, sample_rate=16000, 
                stride=1, padding=0, dilation=1, min_low_hz=50, min_band_hz=50, in_channels=1):
        """
        Args:
            out_channels : `int` number of filters.
            kernel_size : `int` filter length.
            sample_rate : `int`, optional sample rate. Defaults to 16000.
        """
        super(SincNetFilterConvLayer, self).__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only support one input channel (here, in_channels = {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size % 2 == 0:
            # Forcing the filters to be odd (i.e, perfectly symmetrics)
            self.kernel_size += 1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            2595 * np.log10(1 + low_hz / 700),
            2595 * np.log10(1 + high_hz / 700),
            self.out_channels + 1
        )
        hz = 700 * (10 ** (mel / 2595) - 1)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1)) # filter lower frequency (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1)) # filter frequency band (out_channels, 1)

        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_ = nn.Parameter(0.54 - 0.46 * torch.cos(2*pi*n_lin/self.kernel_size))

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = nn.Parameter(2*pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate) # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms : (batch_size, 1, n_samples) batch of waveforms.
        
        Returns:
            features : (batch_size, out_channels, n_samples_out) batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band = (high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])    
        
        band_pass = torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        band_pass = band_pass / (2*band[:,None])
        
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
        ).abs_() # https://github.com/mravanelli/SincNet/issues/4

# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/blocks/sincnet.py
class SincNet(nn.Module):
    """SincNet"""

    def __init__(
        self,  
        num_sinc_filters: int = 80,
        sinc_filter_length: int = 251,
        num_conv_filters: int = 60,
        conv_filter_length: int = 5,
        pool_kernel_size: int = 3,
        pool_stride: int = 3,
        sample_rate: int = 16000,
        stride: int = 10,
    ):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")
            # TODO: add support for other sample rate. it should be enough to multiply
            # sinc_filter_length by (sample_rate / 16000). but this needs to be double-checked.

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList([
            SincNetFilterConvLayer(num_sinc_filters, sinc_filter_length, sample_rate=sample_rate, stride=stride),
            nn.Conv1d(num_sinc_filters, num_conv_filters, conv_filter_length),
            nn.Conv1d(num_conv_filters, num_conv_filters, conv_filter_length),
        ])
        self.pool1d = nn.ModuleList([
            nn.MaxPool1d(pool_kernel_size, stride=pool_stride),
            nn.MaxPool1d(pool_kernel_size, stride=pool_stride),
            nn.MaxPool1d(pool_kernel_size, stride=pool_stride),
        ])
        self.norm1d = nn.ModuleList([
            nn.InstanceNorm1d(num_sinc_filters, affine=True),
            nn.InstanceNorm1d(num_conv_filters, affine=True),
            nn.InstanceNorm1d(num_conv_filters, affine=True),
        ])

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms : (batch, channel, sample_rate)
        """
        outputs = self.wav_norm1d(waveforms)

        for _, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):
            outputs = conv1d(outputs)
            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs

class SincNetModel(PreTrainedModel):
    config_class = SincNetConfig
    base_model_prefix = "sincnet"

    def __init__(self, config: SincNetConfig):
        super().__init__(config)

        self.model = SincNet(
            stride=config.stride,
            num_sinc_filters=config.num_sinc_filters,
            sinc_filter_length=config.sinc_filter_length,
            num_conv_filters=config.num_conv_filters,
            conv_filter_length=config.conv_filter_length,
            pool_kernel_size=config.pool_kernel_size,
            pool_stride=config.pool_stride,
            sample_rate=config.sample_rate,
        )
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.model(waveforms)
