from pydantic import BaseModel


class LLMConfig(BaseModel):
    num_layers: int
    num_heads: int
    d_model: int
    vocab_size: int = 50257
    ctx_len: int = 1024
    d_ff: int = 6400
    rope_theta: float = 1e5


def num_params(c: LLMConfig):
    n = c.vocab_size * c.d_model  # Embedding table
    for _ in range(c.num_layers):
        n += 2 * c.d_model  # 2 pre-RMSNorms
        n += 4 * c.d_model**2  # QKV + out proj weight
        n += 3 * c.d_ff * c.d_model  # ffn with swiglu
    n += c.d_model  # last RMSNorm of attn output
    n += c.vocab_size * c.d_model  # LM head proj

    print(f"Model size: {n / 1e9:.2f}B")


def num_flops(c: LLMConfig):
    batch = c.ctx_len

    attn = (7 + 2 / c.num_heads) * batch * c.d_model**2 * c.num_layers
    ffwd = 20 * batch * c.d_model**2 * c.num_layers
    pred = batch * c.d_model * c.vocab_size
    total = attn + ffwd + pred

    print(f"Attn   : {attn / total * 100:2.0f} %")
    print(f"FFN    : {ffwd / total * 100:2.0f} %")
    print(f"LM Head: {pred / total * 100:2.0f} %")
    print(f"Total  : {total / 1e12:.2f} TFLOPs")


if __name__ == "__main__":
    configs = {
        "s": LLMConfig(
            num_layers=12,
            d_model=768,
            num_heads=12,
        ),
        "m": LLMConfig(
            num_layers=24,
            d_model=1024,
            num_heads=16,
        ),
        "l": LLMConfig(
            num_layers=36,
            d_model=1280,
            num_heads=20,
        ),
        "xl": LLMConfig(
            num_layers=48,
            d_model=1600,
            num_heads=25,
        ),
    }

    for size in ["s", "m", "l", "xl"]:
        conf = configs[size]
        print("Model size", size)
        num_params(conf)
        num_flops(conf)
        print()
