import os
import time
from pathlib import Path

import numpy as np
import tiktoken

from cs336_basics.bpe import make_chunks

CHUNK_SIZE = 4096 * 4096  # 16 MB


def tokenize_data(fpath: str) -> None:
    enc = tiktoken.get_encoding("gpt2")
    savepath = Path(fpath).parent / f"{Path(fpath).stem}.dat"

    fin = open(fpath, "rb")
    fout = open(savepath, "wb")

    fin.seek(0, os.SEEK_END)
    fin_size = fin.tell()
    num_chunks = (fin_size - 1) // CHUNK_SIZE + 1

    boundaries = make_chunks(fin, num_chunks)
    batch_size = os.cpu_count()
    if batch_size is None:
        raise RuntimeError("num_threads must be an integer")

    num_finish = 0
    print(f"Encoding {fpath}...")

    for i in range(0, num_chunks, batch_size):
        t = time.perf_counter()
        batch: list[str] = []
        for j in range(batch_size):
            if i + j + 1 < len(boundaries):
                si, ei = boundaries[i + j], boundaries[i + j + 1]
                fin.seek(si)
                chunk_b = fin.read(ei - si)
                chunk_t = chunk_b.decode("utf-8")
                batch.append(chunk_t)

        outs = enc.encode_batch(batch, num_threads=batch_size, allowed_special={"<|endoftext|>"})
        for out in outs:
            arr = np.array(out, dtype=np.uint16)
            arr.tofile(fout)

        num_finish += len(outs)
        print(f"\tfinish {num_finish}/{num_chunks}: {time.perf_counter() - t:.2f} s", end="\r")
    
    print()
    fin.close()
    fout.close()


if __name__ == "__main__":
    tokenize_data("data/owt_train.txt")
    tokenize_data("data/owt_valid.txt")
    tokenize_data("data/TinyStoriesV2-GPT4-train.txt")
    tokenize_data("data/TinyStoriesV2-GPT4-valid.txt")
