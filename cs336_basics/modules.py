import torch
from torch import nn, Tensor
from einops import einsum, rearrange


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
