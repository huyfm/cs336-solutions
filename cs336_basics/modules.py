import torch
from einops import einsum, reduce
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
        self.W = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


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
        mean_squared = reduce(x**2, "b t d_model -> b t 1", "mean")
        rms = (mean_squared + self.eps) ** 0.5
        return x / rms * self.gain


class SwiGLUFeedFoward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # first linear projection before SwiGLU.
        self.l1 = Linear(d_model, d_ff)
        # linear projection used inside SwiGLU.
        self.l3 = Linear(d_model, d_ff)
        # second lineare projection.
        self.l2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.l1(x)
        y2 = self.l3(x)
        silu_out = torch.sigmoid(y1) * y1
        swiglu_out = silu_out * y2
        return self.l2(swiglu_out)
