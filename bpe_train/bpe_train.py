from cs336_basics.tokenizer import serialize_bpe, train_bpe


def main():
    trainpath = "data/TinyStoriesV2-GPT4-train.txt"
    respath = "bpe_train/tinystories"
    vocab, merges = train_bpe(trainpath, vocab_size=10000, special_tokens=["<|endoftext|>"])
    serialize_bpe(respath, vocab, merges)


if __name__ == "__main__":
    main()
