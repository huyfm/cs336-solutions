import math

import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.emb_table = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        nn.init.trunc_normal_(self.emb_table, mean=0, std=1, a=-3, b=3)

    def forward(self, x: Tensor) -> Tensor:
        return self.emb_table[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f32 = x.to(torch.float32)
        # Use f32 to avoid overflow when squaring.
        mean_squared = reduce(x_f32**2, "b t d_model -> b t 1", "mean")
        rms = torch.sqrt(mean_squared + self.eps)
        return x / rms * self.gain


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.l1 = Linear(d_model, d_ff)
        self.l2 = Linear(d_model, d_ff)
        self.l3 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.l1(x)
        y2 = self.l2(x)
        swiGLU_x = torch.sigmoid(y1) * y1 * y2
        return self.l3(swiGLU_x)


class RoPE(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        base_angle = theta ** (-torch.arange(0, d_k, 2) / d_k)
        pos = torch.arange(max_seq_len)
        phi = einsum(pos, base_angle, "i, j -> i j")  # (maxseq, dpair)

        self.register_buffer("sin_phi", torch.sin(phi).to(device))  # (maxseq, dpair)
        self.register_buffer("cos_phi", torch.cos(phi).to(device))  # (maxseq, dpair)

    def forward(
        self, x: Float[Tensor, "... seq d"], token_positions: Int[Tensor, "... seq"]
    ) -> Float[Tensor, "... seq d"]:
        if torch.max(token_positions) >= self.max_seq_len:
            raise ValueError("There exists positions > max_seq_len")

        xpair = rearrange(x, "... seq (d1 d2) -> ... seq d1 d2", d2=2)
        x1, x2 = xpair[..., 0], xpair[..., 1]  # (..., seq, dpair)

        sin_phi = self.sin_phi[token_positions]  # type: ignore # (..., seq, dpair)
        cos_phi = self.cos_phi[token_positions]  # type: ignore # (..., seq, dpair)
        x1rot = cos_phi * x1 - sin_phi * x2
        x2rot = sin_phi * x1 + cos_phi * x2

        xrot = rearrange([x1rot, x2rot], "b ... seq dpair -> ... seq (dpair b)")
        return xrot
