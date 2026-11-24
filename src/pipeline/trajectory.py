import numpy as np
from src.core import state


def transform(self, x):
    z = np.clip(state.config.beta + state.config.gamma * x, -1.0, 1.0)
    y = state.config.alpha * np.arccos(z) + state.config.zeta
    y = np.clip(y, state.config.low_sigma, state.config.high_sigma)
    return y



def cfg_cdf(x, cfg_scale) -> np.ndarray:
    z = np.clip(state.config.cfg_beta - state.config.cfg_gamma * x, -1.0, 1.0)
    y = state.config.cfg_alpha * np.arccos(z) - state.config.cfg_zeta
    y = np.clip(y, state.config.low_sigma, state.config.high_sigma)

    return y * cfg_scale
