import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from io import BufferedReader

import regex as re

REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DELIMITER = b"<|endoftext|>"
NUM_PROCS = mp.cpu_count()

BytePair = tuple[bytes, bytes]


class PreToken:
    def __init__(self, s: str):
        # Pre-compute hash to avoid rehashing many times.
        # Use md5 instead of default string hash so that
        # hashing in all processes produces the same hash.
        hashstr = hashlib.md5(bytes(s, "utf-8")).hexdigest()
        self.hash = int(hashstr, 16)
        # Current token list in the PreToken.
        # It wil be updated by handle_pretoken function.
        self.tokens = [bytes([b]) for b in bytes(s, "utf-8")]

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other) -> bool:
        return isinstance(other, PreToken) and self.hash == other.hash

    def __repr__(self) -> str:
        return f"PreToken<{self.tokens}>"


def pretokenize(text: str, pat: str, special_tokens: list[str]) -> Counter[PreToken]:
    counts = Counter()
    delim = "|".join(re.escape(s) for s in special_tokens)

    # Split text on special tokens, count pretokens in each split.
    for split in re.splititer(delim, text):
        for match in re.finditer(pat, split):
            p = match.group()
            counts[p] += 1

    # Replace pretoken strings with PreToken objects.
    res = Counter()
    for p in counts:
        res[PreToken(p)] = counts[p]
    return res


def _pretokenize_worker(fpath: str, si: int, ei: int, pat: str, special_tokens: list[str], q: mp.Queue) -> None:
    file = open(fpath, "rb")
    file.seek(si)
    text = file.read(ei - si).decode("utf-8")
    counts = pretokenize(text, pat, special_tokens)
    q.put(counts)
    file.close()


def mp_pretokenize(filepath: str, special_tokens: list[str]) -> Counter[PreToken]:
    file = open(filepath, "rb")
    nprocs = NUM_PROCS
    boundaries = make_chunks(file, nprocs, DELIMITER)
    file.close()

    procs: list[mp.Process] = []
    q = mp.Queue()
    for i in range(nprocs):
        si, ei = boundaries[i : i + 2]
        p = mp.Process(target=_pretokenize_worker, args=(filepath, si, ei, REGEX_PATTERN, special_tokens, q))
        procs.append(p)
        p.start()

    # Main process reads right after processes start to avoid blocking queue -> deadlock.
    pretoken_counts: Counter[PreToken] = Counter()
    for _ in range(nprocs):
        pretoken_counts.update(q.get())

    # Main process waits unitl all child processes to finish.
    for p in procs:
        p.join()

    return pretoken_counts


def make_chunks(file: BufferedReader, num_chunks: int, delimiter: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    approx_chunk_size = file_size // num_chunks

    # Initial chunk boundaries: chunk ith is in range [boundaries[i], boundaries[i+1])
    boundaries = [approx_chunk_size * i for i in range(num_chunks)]
    boundaries.append(file_size)

    buf_size = 4096 * 4

    # Correct boundaries to respect the delimiter.
    for i in range(num_chunks - 1):
        end_idx = boundaries[i + 1]
        file.seek(end_idx)
        buf = file.read(buf_size)
        delim_idx = buf.find(delimiter)
        # If delimiter is found, shift the right boundary to it.
        if delim_idx >= 0:
            boundaries[i + 1] = end_idx + delim_idx
        # If delimiter is not found, find the next space character
        # and shift the right boundary to it to avoid chunking at the
        # middle byte of a utf-8 character.
        else:
            space_idx = buf.find(b" ")
            boundaries[i + 1] = end_idx + space_idx

    return boundaries


def init_global_bpcount(pretokens: Counter[PreToken]) -> Counter[BytePair]:
    """Initialize global byte pair count."""
    bpcount: Counter[BytePair] = Counter()
    for p in pretokens.keys():
        for b1, b2 in zip(p.tokens[:-1], p.tokens[1:]):
            bp = (b1, b2)
            bpcount[bp] += pretokens[p]
    return bpcount


def init_bpe(special_tokens: list[str]) -> tuple[dict[int, bytes], list[BytePair]]:
    merges: list[BytePair] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, s in enumerate(special_tokens):
        vocab[256 + i] = bytes(s, "utf-8")
    return vocab, merges


def train_bpe(filepath: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[BytePair]]:
    t0 = time.perf_counter()
    pretokens = mp_pretokenize(filepath, special_tokens)
    print(f"Pretokenization done: {since(t0):.2f} s")

    bpcount = init_global_bpcount(pretokens)
    vocab, merges = init_bpe(special_tokens)

    # Merge the most frequent pair of bytes until reach vocab size.
    print("BPE initialization done. Start BPE merging...")
    nruns = vocab_size - len(vocab)
    for i in range(nruns):
        t0 = time.perf_counter()
        merge_step(pretokens, bpcount, vocab, merges)
        print(f"\tmerge {i}/{nruns} done: {since(t0):.2f} s")

    return vocab, merges


def merge_step(
    pretokens: Counter[PreToken], bpcount: Counter[BytePair], vocab: dict[int, bytes], merges: list[BytePair]
) -> None:
    # 1. Get most frequent byte pair.
    max_bp = max(bpcount, key=lambda x: (bpcount[x], x))

    # 2. Add the most frequent byte pair to merges and vocab.
    merges.append(max_bp)
    vocab[len(vocab)] = b"".join(max_bp)

    # 3. Update global byte pair count bpcount by processing each pretoken
    # and update bpcount in-place for all pairs that are affected by the merge.
    # And also update the token list in the pretoken with the new merged token.
    for p, pcount in pretokens.items():
        handle_pretoken(p, pcount, max_bp, bpcount)


def handle_pretoken(p: PreToken, pcount: int, merged_bp: BytePair, bpcount: Counter[BytePair]) -> None:
    new_tokens = []
    merged_token = b"".join(merged_bp)
    i = 0
    while i < len(p.tokens):
        # Either reach the end or current pair not match
        if i == len(p.tokens) - 1 or tuple(p.tokens[i : i + 2]) != merged_bp:
            new_tokens.append(p.tokens[i])
            i += 1
            continue

        # Found a match, reduce merged pair count.
        new_tokens.append(merged_token)
        bpcount[merged_bp] -= pcount

        # Previous pair will be replaced by a pair with the second token
        # becomes the merged token.
        if i - 1 >= 0:
            prev_bp = (p.tokens[i - 1], p.tokens[i])
            new_bp = (p.tokens[i - 1], merged_token)
            bpcount[prev_bp] -= pcount
            bpcount[new_bp] += pcount

        # Next pair will be replaced by a pair with the first token
        # becomes the merged token.
        if i + 2 < len(p.tokens):
            next_bp = (p.tokens[i + 1], p.tokens[i + 2])
            new_bp = (merged_token, p.tokens[i + 2])
            bpcount[next_bp] -= pcount
            bpcount[new_bp] += pcount

        i += 2

    p.tokens = new_tokens


def since(t0: float) -> float:
    return time.perf_counter() - t0  # in seconds


def btos(bs: bytes) -> str:
    """Convert UTF-8 bytes to string, replace invalid bytes with equivalent hex strings."""
    return str(bs)[2:-1]


def serialize_bpe(dirpath: str, vocab: dict[int, bytes], merges: list[BytePair]) -> None:
    # Convert bytes in vocab to string as bytes are not JSON serializable
    out_vocab: dict[str, int] = {}
    for i, bs in vocab.items():
        out_vocab[btos(bs)] = i

    with open(f"{dirpath}/bpe_vocab.json", "w") as f:
        json.dump(out_vocab, f, indent=2)

    with open(f"{dirpath}/bpe_merges.txt", "w") as f:
        for b1, b2 in merges:
            line = btos(b1) + " " + btos(b2) + "\n"
            f.write(line)


def test_mp_pretokenizer_sanity_check():
    counts = mp_pretokenize("data/tinystories_sample_5M.txt", special_tokens=["<|endoftext|>"])
    total_count = sum(counts.values())
    wcount = 1033872
    assert total_count > 0.95 * wcount, f"Pretoken count should roughly equal word count"


def main():
    trainpath = "data/tinystories_sample_5M.txt"
    respath = "bpe_train/"
    vocab, merges = train_bpe(trainpath, vocab_size=500, special_tokens=["<|endoftext|>"])
    serialize_bpe(respath, vocab, merges)


if __name__ == "__main__":
    main()
