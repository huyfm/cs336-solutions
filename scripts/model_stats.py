from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: int
    num_layers: int
    num_heads: int
    ctx_len: int
    d_model: int
    d_ff: int
    rope_theta: float = 1e5


def num_params(c: ModelConfig):
    count = 0
    count += c.vocab_size * c.d_model  # Embedding table

    for i in range(c.num_layers):
        count += 2 * c.d_model  # 2 pre-RMSNorms
        count += 4 * c.d_model**2  # QKV + out proj weight
        count += 3 * c.d_ff * c.d_model  # ffn with swiglu

    count += c.d_model  # last RMSNorm of attn output
    count += c.vocab_size * c.d_model  # LM head proj

    nparams = count / 1e9
    print(f"Model size: {nparams:.2f}B")

    return nparams


def num_flops(c: ModelConfig):
    common = 2 * c.ctx_len * c.d_model  # common factor
    count = 0

    # Forwarding the blocks.
    each_block = 3 + c.d_model / c.num_heads + c.ctx_len + c.d_model + 3 * c.d_ff
    count += c.num_layers * each_block
    # Forward LM head.
    count += c.vocab_size

    nflops = common * count / 1e12
    print(f"Forward Comp: {nflops:.2f}FLOPs")
    print("Proportion:")
    print(f"\tEach block: {each_block / count * 100:.2f}%")
    print(f"\tLM head   : {c.vocab_size / count * 100:.2f}%")

    return nflops


if __name__ == "__main__":
    configs = {
        "gpt2_xl": ModelConfig(
            vocab_size=50257,
            ctx_len=1024,
            num_layers=48,
            d_model=1600,
            num_heads=25,
            d_ff=6400,
        )
    }

    config_xl = configs["gpt2_xl"]

    num_params(config_xl)
    num_flops(config_xl)
