import numpy as np
from src.core import Config

def sigma_cdf(x, config:Config) -> np.ndarray:
    z = np.clip(config.beta + config.gamma * x, -1.0, 1.0)
    y = config.alpha * np.arccos(z) + config.zeta
    y = np.clip(y, config.low_sigma, config.high_sigma)

    return y

def cfg_cdf(x, cfg_scale, config:Config) -> np.ndarray:
    z = np.pi*(config.cfg_alpha + config.cfg_beta * x) / config.cfg_gamma
    y = np.sin(z)
    y = np.clip(y, 0, 1)

    return y * cfg_scale