import math

import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Bool, Float, Int
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
        mean_squared = reduce(x_f32**2, "b seq d_model -> b seq 1", "mean")
        rms = torch.sqrt(mean_squared + self.eps)
        return x / rms * self.gain


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.fc1 = Linear(d_model, 2 * d_ff, dtype, device)
        self.fc2 = Linear(d_ff, d_model, dtype, device)

    def forward(self, x: Tensor) -> Tensor:
        y = rearrange(self.fc1(x), "... (d2 d_ff) -> ... d2 d_ff", d2=2)
        y1 = y[..., 0, :]
        y2 = y[..., 1, :]
        swiglu_x = torch.sigmoid(y1) * y1 * y2
        return self.fc2(swiglu_x)


class RoPE(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        base_angle = theta ** (-torch.arange(0, d_k, 2) / d_k)
        pos = torch.arange(max_seq_len)
        phi = einsum(pos, base_angle, "i, j -> i j")  # (maxseq, dpair)

        # Both buffers of size (maxseq, dpair).
        self.register_buffer("sin_phi", torch.sin(phi).to(device), persistent=False)
        self.register_buffer("cos_phi", torch.cos(phi).to(device), persistent=False)

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

        xrot = rearrange([x1rot, x2rot], "b ... seq dpair -> ... seq (dpair b)")  # (... seq, d)
        return xrot


def softmax(x: Tensor, dim: int) -> Tensor:
    xmax = torch.max(x, dim=dim, keepdim=True).values
    x = x - xmax  # for numerical reason
    xexp = torch.exp(x)
    prob = xexp / torch.sum(xexp, dim=dim, keepdim=True)
    return prob


def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_q d_k"],
    K: Float[Tensor, "... seq_k d_k"],
    V: Float[Tensor, "... seq_k d_v"],
    mask: Bool[Tensor, " ... seq_q seq_k"] | None = None,
) -> Float[Tensor, " ... seq_q d_v"]:
    d_k = Q.size(-1)
    attn_scores = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / math.sqrt(d_k)
    # Replace false mask values with -inf.
    if not mask is None:
        attn_scores.masked_fill_(~mask, float("-inf"))

    attn_weights = softmax(attn_scores, dim=-1)  # (..., seq_q, seq_k)
    output = einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return output


class CausalMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        enable_rope: bool,
        theta: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.h = num_heads
        self.max_seq_len = max_seq_len
        self.enable_rope = enable_rope
        d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3 * d_model, dtype, device)
        self.out_proj = Linear(d_model, d_model, dtype, device)

        if enable_rope:
            if theta is None:
                raise ValueError("Theta cannot be None if RoPE is enabled")
            self.rope = RoPE(theta, d_k, max_seq_len, device)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("mask", mask, persistent=False)  # (maxseq, maxseq)

    def forward(self, x: Float[Tensor, "b seq d_model"]) -> Float[Tensor, "b seq d_model"]:
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError("Input sequence length > max_seq_len")

        QKV = rearrange(self.qkv_proj(x), "b seq (d2 d_model) -> b seq d2 d_model", d2=3)
        Q = rearrange(QKV[..., 0, :], "b seq (h d_k) -> b h seq d_k", h=self.h)
        K = rearrange(QKV[..., 1, :], "b seq (h d_k) -> b h seq d_k", h=self.h)
        V = rearrange(QKV[..., 2, :], "b seq (h d_k) -> b h seq d_k", h=self.h)

        if self.enable_rope:
            pos_ids = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, pos_ids)
            K = self.rope(K, pos_ids)

        mask = self.mask[:seq_len, :seq_len]  # type: ignore
        attn = scaled_dot_product_attention(Q, K, V, mask)  # (b, h, seq, d_k)
        attn = rearrange(attn, "b h seq d_k -> b seq (h d_k)")  # (b, seq, d_model)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.rmsn1 = RMSNorm(d_model, 1e-5, dtype, device)
        self.rmsn2 = RMSNorm(d_model, 1e-5, dtype, device)
        enabled_rope = True
        self.attn = CausalMHA(d_model, num_heads, max_seq_len, enabled_rope, theta, dtype, device)
        self.ffn = FFN(d_model, d_ff, dtype, device)

    def forward(self, x: Float[Tensor, "b seq d_model"]) -> Float[Tensor, "b seq d_model"]:
        xnorm = self.rmsn1(x)
        x = x + self.attn(xnorm)
        xnorm = self.rmsn2(x)
        x = x + self.ffn(xnorm)
        return x
