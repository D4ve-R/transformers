""" SincNet model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

SINCNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "sincnet-base": "https://huggingface.co/sincnet-base/resolve/main/config.json",
}

class SincNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transform
    ers.SincNetModel`. It is used to instantiate a SincNet model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        sample_rate (:obj:`int`, `optional`, defaults to 16000):
            Audio sample rate.
        stride (:obj:`int`, `optional`, defaults to 10):
            Stride for the SincNet filter block.
        num_sinc_filters (:obj:`int`, `optional`, defaults to 80):
            Number of sinc filters.
        sinc_filter_length (:obj:`int`, `optional`, defaults to 251):
            Sinc filter length.
        num_conv_filters (:obj:`int`, `optional`, defaults to 60):
            Number of convolutional filters.
        conv_filter_length (:obj:`int`, `optional`, defaults to 5):
            Convolutional filter length.
        pool_kernel_size (:obj:`int`, `optional`, defaults to 3):
            Pool kernel size.
        pool_stride (:obj:`int`, `optional`, defaults to 3):
            Pool stride.

    Example:

    ```python
    >>> from transformers import SincNetModel, SincNetConfig

    >>> # Initializing a SincNet configuration
    >>> configuration = SincNetConfig()

    >>> # Initializing a model from the configuration
    >>> model = SincNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    model_type = "sincnet"
    
    def __init__(
        self,
        sample_rate: int = 16000,
        stride: int = 10,
        num_sinc_filters: int = 80,
        sinc_filter_length: int = 251,
        num_conv_filters: int = 60,
        conv_filter_length: int = 5,
        pool_kernel_size: int = 3,
        pool_stride: int = 3,
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.stride = stride
        self.num_sinc_filters = num_sinc_filters
        self.sinc_filter_length = sinc_filter_length
        self.num_conv_filters = num_conv_filters
        self.conv_filter_length = conv_filter_length
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        super().__init__(**kwargs)   
