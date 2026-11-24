import numpy as np
from src.core import state

def sigma_cdf(x) -> np.ndarray:
    z = np.clip(state.config.beta - state.config.gamma * x, -1.0, 1.0)
    y = state.config.alpha * np.arccos(z) - state.config.zeta
    y = np.clip(y, state.config.low_sigma, state.config.high_sigma)

    return y


def cfg_cdf(x, cfg_scale) -> np.ndarray:
    z = np.pi*(state.config.cfg_beta + state.config.cfg_gamma * x) / state.config.cfg_zeta
    y = state.config.cfg_alpha * np.sin(z)
    y = np.clip(y, state.config.low_sigma, state.config.high_sigma)

    return y * cfg_scale

steps = 100

u = np.linspace(0.0, 1.0, steps + 1)
sigmas = sigma_cdf(u)[::-1].copy()

c = np.linspace(0.0, 1.0, steps + 1)
cfgs = cfg_cdf(c, 4.0)[::-1].copy()

print(f"len(sigmas)={len(sigmas)}, len(cfgs)={len(cfgs)}")
print(f"{'step':>4} | {'sigma_cur':>10} | {'sigma_next':>10} | {'cfg':>8}")
print("-" * 42)

for i in range(steps):
    print(f"{i:>4} | {sigmas[i]:>10.5f} | {sigmas[i+1]:>10.5f} | {cfgs[i]:>8.4f}")