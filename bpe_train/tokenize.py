import numpy as np

from cs336_basics.tokenizer import Tokenizer

BUFSIZE = 4096


def get_encoder(name: str) -> Tokenizer:
    if name == "openwebtext":
        return Tokenizer.from_files(
            vocab_filepath="bpe_train/openwebtext/bpe_vocab.pkl",
            merges_filepath="bpe_train/openwebtext/bpe_merges.pkl",
            special_tokens=["<|endoftext|>"],
        )
    if name == "tinystories":
        return Tokenizer.from_files(
            vocab_filepath="bpe_train/tinystories/bpe_vocab.pkl",
            merges_filepath="bpe_train/tinystories/bpe_merges.pkl",
            special_tokens=["<|endoftext|>"],
        )
    raise ValueError("Supported encoder name: openwebtext, tinystories")


def tokenize_tinystories() -> None:
    enc = get_encoder("tinystories")
    out = open("data/TinyStoriesV2-GPT4-valid.dat", "wb")
    buf = np.zeros(4096, dtype=np.float16)
    i = 0  # buffer's write index

    f = open("data/TinyStoriesV2-GPT4-valid.txt")
    enc.encode(f.read())
    print("DONE")
    # for idx in enc.encode_iterable(f):
    #     buf[i] = idx
    #     i += 1
    #     if i >= BUFSIZE:
    #         buf.tofile(out)
    #         i = 0

    # out.close()
    # f.close()


if __name__ == "__main__":
    tokenize_tinystories()
