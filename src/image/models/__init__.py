"""
Generative and discriminative image models.
"""

from .dcgan import DCGANConfig, train_dcgan
from .ddpm import DDPMConfig, train_ddpm

__all__ = ["DCGANConfig", "train_dcgan", "DDPMConfig", "train_ddpm"]
