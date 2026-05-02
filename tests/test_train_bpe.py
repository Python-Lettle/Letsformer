from letsformer import train_bpe
import pickle

DATASET_PATH = "./data/"

def test_train_bpe():
    VOCAB_SIZE = 500
    SPECIAL_TOKENS = ["<|endoftext|>"]

    data_path = DATASET_PATH + "low_data.txt"

    vocab, merges = train_bpe(data_path, VOCAB_SIZE, SPECIAL_TOKENS)

    assert len(vocab) == 273
    assert len(merges) == 16

    pickle.dump(vocab, open("data/tokenizer/vocab.pkl", "wb"))
    pickle.dump(merges, open("data/tokenizer/merges.pkl", "wb"))

    print("vocab 和 merges 已经保存至 vocab.pkl 和 merges.pkl")
