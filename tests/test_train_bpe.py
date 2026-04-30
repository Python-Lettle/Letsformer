from letsformer import train_bpe

DATASET_PATH = "./data/"

def test_train_bpe():
    VOCAB_SIZE = 500
    SPECIAL_TOKENS = ["<|endoftext|>"]

    data_path = DATASET_PATH + "low_data.txt"

    vocab, merges = train_bpe(data_path, VOCAB_SIZE, SPECIAL_TOKENS)

    assert isinstance(vocab, dict)
    assert isinstance(merges, list)

    print(vocab)
    print(merges)
