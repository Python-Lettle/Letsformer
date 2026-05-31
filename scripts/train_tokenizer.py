'''
    训练 tokenizer
'''
from letsformer import train_bpe
from letsformer.debug import console, DEBUG
import pickle
import time

def train_tokenizer():
    DATA_FILE = "./data/low_data.txt"

    VOCAB_SIZE = 10000
    SPECIAL_TOKENS = ["<|endoftext|>"]

    start_time = time.time()
    vocab, merges = train_bpe(
        DATA_FILE,
        VOCAB_SIZE,
        SPECIAL_TOKENS,
    )
    end_time = time.time()
    console.print(f"(train_tokenizer) Tokenizer trained in {end_time - start_time} seconds.")
    console.print(f"(train_tokenizer) Vocabulary size: {len(vocab)}")
    console.print(f"(train_tokenizer) Merges size: {len(merges)}")

    TOKENIZER_ROOT = "./data/tinystories_valid_tokenizer/"
    with open(TOKENIZER_ROOT + "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    with open(TOKENIZER_ROOT + "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    if DEBUG:
        console.print("(train_tokenizer) Tokenizer trained and saved.")

if __name__ == "__main__":
    train_tokenizer()