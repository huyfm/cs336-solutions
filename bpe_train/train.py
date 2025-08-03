from cs336_basics.tokenizer import serialize_bpe, train_bpe


def main():
    # trainpath = "data/TinyStoriesV2-GPT4-train.txt"
    # respath = "bpe_train/tinystories"

    trainpath = "data/owt_train.txt"
    respath = "bpe_train/openwebtext"

    vocab, merges = train_bpe(trainpath, vocab_size=32000, special_tokens=["<|endoftext|>"])
    serialize_bpe(respath, vocab, merges)


if __name__ == "__main__":
    main()
