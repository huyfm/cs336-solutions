import regex as re
from collections import Counter

REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

BytePair = tuple[bytes, bytes]


class PreToken:
    def __init__(self, s: str):
        # Pre-compute hash to avoid rehashing many times.
        self.hash = hash(s)
        # Current token list in the PreToken.
        # It wil be updated by handle_pretoken function.
        self.tokens = [bytes([b]) for b in bytes(s, "utf-8")]

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self) -> str:
        return f"PreToken<{self.tokens}>"


def pretokenize(text: str, pat: str) -> Counter[PreToken]:
    # Count pretokens in text.
    counts = Counter()
    for match in re.finditer(pat, text):
        p = match.group()
        counts[p] += 1

    # Replace pretoken strings with PreToken objects.
    res = Counter()
    for p in counts:
        res[PreToken(p)] = counts[p]
    return res


def init_global_bpcount(pretokens: Counter[PreToken]) -> Counter[BytePair]:
    """Initialize global byte pair count."""
    bpcount: Counter[BytePair] = Counter()
    for p in pretokens.keys():
        for b1, b2 in zip(p.tokens[:-1], p.tokens[1:]):
            bp = (b1, b2)
            bpcount[bp] += pretokens[p]
    return bpcount


def init_bpe():
    merges: list[BytePair] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(1, 256)}
    vocab[0] = b"<|endoftext|>"
    return vocab, merges


def train_bpe(text: str, vocab_size: int, special_tokens: list[str]):
    pretokens = pretokenize(text, REGEX_PATTERN)
    bpcount = init_global_bpcount(pretokens)
    vocab, merges = init_bpe()

    for _ in range(vocab_size - len(vocab)):
        merge_step(pretokens, bpcount, vocab, merges)

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


def test_main():
    vocab, merges = train_bpe(text, vocab_size=262, special_tokens=[])
    print(vocab)
    print(merges)
