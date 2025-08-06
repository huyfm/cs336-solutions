from pathlib import Path

import numpy as np
import tiktoken

from cs336_basics.bpe import Tokenizer


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
    raise ValueError("Unsupported name")


def tokenize_data(fpath: str) -> None:
    enc = tiktoken.get_encoding("gpt2")
    savepath = Path(fpath).parent / f"{Path(fpath).stem}.dat"

    with open(fpath, encoding="utf-8") as fin, open(savepath, "ab") as fout:
        for line in fin:
            ids = enc.encode(line, allowed_special={"<|endoftext|>"})
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(fout)


if __name__ == "__main__":
    tokenize_data("data/TinyStoriesV2-GPT4-valid.txt")
