import math
from collections.abc import Callable

import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.optim.optimizer import ParamsT


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        # Use truncated He initialization.
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
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        # Use truncated standard normal distribution to initialize.
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f32 = x.to(torch.float32)
        # Use f32 to avoid overflow when squaring.
        mean_squared = reduce(x_f32**2, "b seq d_model -> b seq 1", "mean")
        rms = torch.sqrt(mean_squared + self.eps)
        return x / rms * self.weight


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # SwiGLU projection.
        self.fc1 = Linear(d_model, 2 * d_ff, dtype, device)
        # projection back to the residual path.
        self.fc2 = Linear(d_ff, d_model, dtype, device)

    def forward(self, x: Tensor) -> Tensor:
        y = rearrange(self.fc1(x), "... (d2 d_ff) -> ... d2 d_ff", d2=2)
        y1 = y[..., 0, :]
        y2 = y[..., 1, :]
        swiglu_x = torch.sigmoid(y1) * y1 * y2
        return self.fc2(swiglu_x)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Compute rotating angles phi.
        base_angle = theta ** (-torch.arange(0, d_k, 2) / d_k)
        pos = torch.arange(max_seq_len)
        phi = einsum(pos, base_angle, "i, j -> i j")  # (maxseq, dpair)

        # Both buffers of size (maxseq, dpair).
        self.register_buffer("sin_phi", torch.sin(phi).to(device, dtype), persistent=False)
        self.register_buffer("cos_phi", torch.cos(phi).to(device, dtype), persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq d"], token_positions: Int[Tensor, "... seq"]
    ) -> Float[Tensor, "... seq d"]:
        if torch.max(token_positions) >= self.max_seq_len:
            raise ValueError("There exists positions > max_seq_len")

        xpair = rearrange(x, "... seq (d1 d2) -> ... seq d1 d2", d2=2)
        x1, x2 = xpair[..., 0], xpair[..., 1]  # (..., seq, dpair)

        # Rotate each pair of elements in the input vectors by phi.
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
            self.rope = RoPE(theta, d_k, max_seq_len, dtype, device)

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
        rope_theta: float,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        enabled_rope = True
        self.ln1 = RMSNorm(d_model, 1e-5, dtype, device)
        self.attn = CausalMHA(d_model, num_heads, max_seq_len, enabled_rope, rope_theta, dtype, device)
        self.ln2 = RMSNorm(d_model, 1e-5, dtype, device)
        self.ffn = FFN(d_model, d_ff, dtype, device)

    def forward(self, x: Float[Tensor, "b seq d_model"]) -> Float[Tensor, "b seq d_model"]:
        xnorm = self.ln1(x)
        x = x + self.attn(xnorm)
        xnorm = self.ln2(x)
        x = x + self.ffn(xnorm)
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # Embed token indices to a sequence of token embeddings.
        self.token_embd = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Attention blocks that process the token embeddings
        # and compute the predicted feature vectors.
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, ctx_len, dtype, device) for _ in range(num_layers)
        )

        # Normalize attention output: due to using pre-norm blocks.
        self.ln_f = RMSNorm(d_model, 1e-5, dtype, device)

        # Prediction LM head.
        self.lm_head = Linear(d_model, vocab_size, dtype, device)

    def forward(self, indices: Float[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab"]:
        x = self.token_embd(indices)  # (b, seq, d_model)
        for l in self.layers:
            x = l(x)  # (b, seq, d_model)
        logit = self.lm_head(self.ln_f(x))  # (b, seq, vocab)
        return logit


def cross_entropy(x: Float[Tensor, "... batch dim"], targets: Int[Tensor, "... batch"]) -> Float[Tensor, ""]:
    # b = total batch dimension.
    x = rearrange(x, "... batch dim -> (... batch) dim")  # (b, dim)
    xmax = reduce(x, "b dim -> b 1", "max")
    logsumexp = (x - xmax).exp().sum(dim=-1).log()  # (b,)
    xtarget = x[torch.arange(x.size(0)), targets]  # (b,)
    xmax = xmax.view(-1)  # (b,)

    out = (xmax + logsumexp - xtarget).mean()
    return out


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: int, betas: tuple[int, int], eps: float, weight_decay: float):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                # Initialize param state if not exist.
                if len(state) == 0:
                    state["t"] = 0  # current timestep
                    state["m"] = torch.zeros_like(p)  # first moment
                    state["v"] = torch.zeros_like(p)  # second moment

                # Advance timestep.
                state["t"] += 1

                # Update moment running averages using in-place ops.
                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Fuse bias corrections into current learning rate.
                t = state["t"]
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Apply weight decay.
                p.data.mul_(1 - lr * weight_decay)

                # Apply Adam update.
                # "denom" tensor can be optimized away if we write
                # a fused kernel that does all computation on the fly.
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-lr_t)

        # Return to conform with Optimizer's default method.
        # Actually no ops here.
        loss = None if closure is None else closure()
        return loss
