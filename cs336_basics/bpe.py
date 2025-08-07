import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from io import BufferedReader

import regex as re

PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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
        return f"PreToken<{b''.join(self.tokens)}>"


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


def mp_pretokenize(filepath: str, special_tokens: list[str], nprocs: int = NUM_PROCS) -> Counter[PreToken]:
    file = open(filepath, "rb")
    boundaries = make_chunks(file, nprocs)
    file.close()

    procs: list[mp.Process] = []
    q = mp.Queue()
    for i in range(nprocs):
        si, ei = boundaries[i : i + 2]
        p = mp.Process(target=_pretokenize_worker, args=(filepath, si, ei, PRETOK_PATTERN, special_tokens, q))
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


def make_chunks(file: BufferedReader, num_chunks: int) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    approx_chunk_size = file_size // num_chunks

    # Initial chunk boundaries: chunk ith is in range [boundaries[i], boundaries[i+1])
    boundaries = [approx_chunk_size * i for i in range(num_chunks)]
    boundaries.append(file_size)

    buf_size = 1024

    # Correct boundaries to respect the pretokens.
    for i in range(num_chunks - 1):
        end_idx = boundaries[i + 1]
        file.seek(end_idx)
        buf = file.read(buf_size)
        # Shift right boundary to the next space character.
        # This strategy works well for GPT2's defined pretokens
        # and current datasets (english).
        space_idx = buf.find(b" ")
        boundaries[i + 1] = end_idx + space_idx

    return boundaries


def init_global_bpcount(pretokens: Counter[PreToken]) -> tuple[Counter[BytePair], defaultdict[BytePair, set[PreToken]]]:
    """Initialize global byte pair count and index."""
    bpcount: Counter[BytePair] = Counter()
    bpindex: defaultdict[BytePair, set[PreToken]] = defaultdict(set)
    for p in pretokens.keys():
        for b1, b2 in zip(p.tokens[:-1], p.tokens[1:]):
            bp = (b1, b2)
            bpcount[bp] += pretokens[p]
            bpindex[bp].add(p)

    return bpcount, bpindex


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

    t0 = time.perf_counter()
    bpcount, bpindex = init_global_bpcount(pretokens)
    vocab, merges = init_bpe(special_tokens)
    print(f"BPE initialization done: {since(t0):.2f} s")

    # Merge the most frequent pair of bytes iteratively until reach vocab size.
    print("Start BPE merging...")
    nruns = vocab_size - len(vocab)
    print(f"Number of pretokens:", len(pretokens))
    print(f"Number of merges:", nruns)
    t0 = time.perf_counter()
    for i in range(nruns):
        t1 = time.perf_counter()
        merge_step(pretokens, bpcount, bpindex, vocab, merges)
        print(f"\tmerge {i}/{nruns} done: {since(t1):.2f} s", end="\r")
    print()
    print(f"BPE training done: {since(t0):.2f} s")

    return vocab, merges


def merge_step(
    pretokens: Counter[PreToken],
    bpcount: Counter[BytePair],
    bpindex: defaultdict[BytePair, set[PreToken]],
    vocab: dict[int, bytes],
    merges: list[BytePair],
) -> None:
    # 1. Get most frequent byte pair.
    max_bp = max(bpcount, key=lambda x: (bpcount[x], x))

    # 2. Add the most frequent byte pair to merges and vocab.
    merges.append(max_bp)
    vocab[len(vocab)] = b"".join(max_bp)

    # 3. Process pretokens affected by the merge.
    # Update bpcount and bpindex in-place.
    for p in bpindex[max_bp].copy():  # copy to avoid in-place update causes runtime error
        pcount = pretokens[p]
        handle_pretoken(p, pcount, max_bp, bpcount, bpindex)

    # Remove merged pair's count and index after the merge.
    del bpindex[max_bp]
    del bpcount[max_bp]


def handle_pretoken(
    p: PreToken,
    pcount: int,
    merged_bp: BytePair,
    bpcount: Counter[BytePair],
    bpindex: defaultdict[BytePair, set[PreToken]],
) -> None:
    new_tokens = []
    merged_token = b"".join(merged_bp)
    i = 0
    while i < len(p.tokens):
        # Either reach the end or current pair not match.
        if i == len(p.tokens) - 1 or tuple(p.tokens[i : i + 2]) != merged_bp:
            new_tokens.append(p.tokens[i])
            i += 1
            continue

        # Found a match.
        new_tokens.append(merged_token)

        # Previous pair will be replaced by a pair with the second token
        # becomes the new merged token.
        if i - 1 >= 0:
            old_bp = (p.tokens[i - 1], p.tokens[i])
            new_bp = (p.tokens[i - 1], merged_token)
            # Update bpcount.
            bpcount[old_bp] -= pcount
            bpcount[new_bp] += pcount
            # Update bpindex.
            try:
                bpindex[old_bp].remove(p)
            except KeyError:
                pass
            bpindex[new_bp].add(p)

        # Next pair will be replaced by a pair with the first token
        # becomes the merged token.
        if i + 2 < len(p.tokens):
            old_bp = (p.tokens[i + 1], p.tokens[i + 2])
            new_bp = (merged_token, p.tokens[i + 2])
            # Update bpcount.
            bpcount[old_bp] -= pcount
            bpcount[new_bp] += pcount
            # Update bpindex.
            try:
                bpindex[old_bp].remove(p)
            except KeyError:
                pass
            bpindex[new_bp].add(p)

        i += 2

    p.tokens = new_tokens


def since(t0: float) -> float:
    return time.perf_counter() - t0  # in seconds


def btos(bs: bytes) -> str:
    """Convert UTF-8 bytes to string, replace invalid bytes with equivalent hex strings."""
    return str(bs)[2:-1]


def serialize_bpe(dirpath: str, vocab: dict[int, bytes], merges: list[BytePair]) -> None:
    os.makedirs(dirpath, exist_ok=True)

    # Save vocab and merge list objects to disk.
    with open(f"{dirpath}/bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"{dirpath}/bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # Convert bytes in vocab to string as bytes are not JSON serializable
    out_vocab: dict[str, int] = {}
    for i, bs in vocab.items():
        out_vocab[btos(bs)] = i

    # Save vocab and merge list in human-readable format.
    with open(f"{dirpath}/bpe_vocab.json", "w") as f:
        json.dump(out_vocab, f, indent=2)

    with open(f"{dirpath}/bpe_merges.txt", "w") as f:
        for b1, b2 in merges:
            line = btos(b1) + " " + btos(b2) + "\n"
            f.write(line)


def test_bpe_training():
    # Use pytest to run the test.
    vocab_size = 256 + 1 + 6
    vocab, _ = train_bpe("tests/fixtures/testtext.txt", vocab_size, ["<|endoftext|>"])
    assert len(vocab) == vocab_size

    expected_merged_tokens = [b"ss", b"st", b"est", b"ow", b"low", b"west"]
    assert all(s in vocab.values() for s in expected_merged_tokens)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[BytePair], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []

        # Prioritize longer tokens first.
        special_tokens.sort(key=len, reverse=True)
        # Delimiter pattern to split the text on special tokens.
        # Use group (...) so that re.split also returns delimiter chunks.
        if special_tokens:
            self.delim_pattern = f"({'|'.join(re.escape(s) for s in special_tokens)})"
        else:
            self.delim_pattern = ""

        self.special_tokens: set[bytes] = set(bytes(s, "utf-8") for s in special_tokens)
        # Add special tokens to the vocabulary.
        for bs in self.special_tokens:
            if not bs in self.vocab.values():
                self.vocab[len(self.vocab)] = bs

        # Reverse vocabulary mapping.
        self.btoi = {bs: i for i, bs in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges: list[BytePair] = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Split on special tokens (special tokens are returned as an isolated chunk)
        # Somehow, using regex to find either special tokens or pretokens idea doesn't work.
        # We must split then find.
        if self.delim_pattern:
            chunks = re.split(self.delim_pattern, text)
        else:
            chunks = [text]

        ids: list[int] = []
        for chunk in chunks:
            bytechunk = bytes(chunk, "utf-8")
            if bytechunk in self.special_tokens:
                ids.append(self.btoi[bytechunk])
                continue
            # Break chunk into pretokens.
            pretokens = [bytes(s, "utf-8") for s in re.findall(PRETOK_PATTERN, chunk)]
            # Tokenize each pretoken and collect token IDs.
            for pt in pretokens:
                encoded_ids = self._merge_pretoken(pt)
                ids.extend(encoded_ids)
        return ids

    def _merge_pretoken(self, pretoken: bytes) -> list[int]:
        tokens = [bytes([b]) for b in pretoken]

        # Apply each of the merges in order.
        for merged_bp in self.merges:
            if len(tokens) < 2:
                break
            temp: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merged_bp:
                    temp.append(merged_bp[0] + merged_bp[1])
                    i += 2
                else:
                    temp.append(tokens[i])
                    i += 1
            tokens = temp

        ids = [self.btoi[t] for t in tokens]
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bs = b"".join(self.vocab[i] for i in ids)
        return bs.decode("utf-8", errors="replace")


def test_tokenizer():
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

    # With special tokens.
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    s = "the cat <|endoftext|> ate"
    encoded_s = [9, 7, 1, 5, 0, 11, 10, 3]
    assert tokenizer.encode(s) == encoded_s
    assert tokenizer.decode(encoded_s) == s

    # Without special tokens.
    tokenizer = Tokenizer(vocab, merges)
    s = "the cat ate"
    encoded_s = [9, 7, 1, 5, 10, 3]
    assert tokenizer.encode(s) == encoded_s
    assert tokenizer.decode(encoded_s) == s
