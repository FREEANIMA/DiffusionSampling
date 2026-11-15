import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, sinusoid_rope_hz, check=False):
        super().__init__()
        self.check = check
        self.sinusoid_rope_hz = sinusoid_rope_hz
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(self.sinusoid_rope_hz) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)

        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.check:
            self.check_sigma(x)

        return emb

    def check_sigma(self, x):
        half_dim = self.dim // 2

        emb = math.log(self.sinusoid_rope_hz) / max(half_dim - 1, 1)
        freq = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb
        )

        x_min = x.float().min().item()
        x_max = x.float().max().item()

        indices = sorted(set([0, half_dim // 2, half_dim - 1]))

        for i in indices:
            f = freq[i].item()
            radians = x_max * f
            cycles = radians / (2 * math.pi)