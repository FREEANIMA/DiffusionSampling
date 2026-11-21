import torch
import torch.nn as nn

class Rope2D(nn.Module):
    def __init__(self, rope_hz, pos_scale, check=False):
        super().__init__()
        self.rope_hz = rope_hz
        self.pos_scale = pos_scale
        self.check = check

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _cos_sin(self, n_pos, inv_freq):
        f = torch.outer(n_pos, inv_freq)
        f = torch.cat((f, f), dim=-1)
        return f.cos()[None, None], f.sin()[None, None]

    def forward(self, q, k, H, W):
        B, nh, N, D = q.shape
        assert N == H * W
        assert D % 4 == 0

        d = D // 2
        dev = q.device

        with torch.autocast(device_type=dev.type, enabled=False):
            inv_freq = 1.0 / (
                self.rope_hz ** (
                    torch.arange(
                        0, d, 2,
                        device=dev,
                        dtype=torch.float32
                    ) / d
                )
            )

            h_idx = (
                torch.arange(H, device=dev, dtype=torch.float32)
                * self.pos_scale
            )[:, None].expand(H, W).reshape(-1)

            w_idx = (
                torch.arange(W, device=dev, dtype=torch.float32)
                * self.pos_scale
            )[None, :].expand(H, W).reshape(-1)

            cos_h, sin_h = self._cos_sin(h_idx, inv_freq)
            cos_w, sin_w = self._cos_sin(w_idx, inv_freq)

        def rot(x):
            xf = x.float()

            x_h, x_w = xf.chunk(2, dim=-1)

            x_h = x_h * cos_h + self._rotate_half(x_h) * sin_h
            x_w = x_w * cos_w + self._rotate_half(x_w) * sin_w

            return torch.cat((x_h, x_w), dim=-1).to(x.dtype)

        q_out = rot(q)
        k_out = rot(k)

        return q_out, k_out