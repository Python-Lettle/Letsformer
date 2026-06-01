'''
    训练 
    
    调用方式: uv run scripts/train_tokenizer.py <data_path> <tokenizer_name>
'''
from letsformer import train_bpe
from letsformer.debug import console, DEBUG
import pickle
import time
import os
import sys

def train_tokenizer(data_path: str | os.PathLike, tokenizer_name: str = "tokenizer", vocab_size: int = 10000, special_tokens: list[str] = ["<|endoftext|>"]):
    start_time = time.time()
    vocab, merges = train_bpe(
        data_path,
        vocab_size,
        special_tokens,
    )
    end_time = time.time()
    console.print(f"(train_tokenizer) Tokenizer trained in {end_time - start_time} seconds.")
    console.print(f"(train_tokenizer) Vocabulary size: {len(vocab)}")
    console.print(f"(train_tokenizer) Merges size: {len(merges)}")

    TOKENIZER_ROOT = "./data/" + tokenizer_name + "/"
    with open(TOKENIZER_ROOT + "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    with open(TOKENIZER_ROOT + "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    if DEBUG:
        console.print("(train_tokenizer) Tokenizer trained and saved.")

if __name__ == "__main__":
    train_tokenizer(data_path=sys.argv[1], tokenizer_name=sys.argv[2])