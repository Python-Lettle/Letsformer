'''
    训练 Tokenizer
    
    调用方式: uv run scripts/train_tokenizer.py [data_path] [tokenizer_name] [--config CONFIG_PATH]
'''
from letsformer import train_bpe
from letsformer.debug import console, DEBUG
from letsformer.config import TrainConfig, DEFAULT_CONFIG
import pickle
import time
import os
import sys
import argparse

def train_tokenizer(data_path: str | os.PathLike, config: TrainConfig = DEFAULT_CONFIG):
    start_time = time.time()
    vocab, merges = train_bpe(
        data_path,
        config.tokenizer.vocab_size,
        config.tokenizer.special_tokens,
    )
    end_time = time.time()
    console.print(f"(train_tokenizer) Tokenizer trained in {end_time - start_time} seconds.")
    console.print(f"(train_tokenizer) Vocabulary size: {len(vocab)}")
    console.print(f"(train_tokenizer) Merges size: {len(merges)}")

    os.makedirs(config.tokenizer.tokenizer_root, exist_ok=True)
    with open(config.tokenizer.merges_path, "wb") as f:
        pickle.dump(merges, f)
    with open(config.tokenizer.vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    if DEBUG:
        console.print("(train_tokenizer) Tokenizer trained and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE Tokenizer")
    parser.add_argument("data_path", nargs="?", help="Path to training data")
    parser.add_argument("tokenizer_name", nargs="?", help="Name of tokenizer directory")
    parser.add_argument("--config", help="Path to config JSON file")
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = TrainConfig.from_json(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # 如果命令行指定了参数，覆盖配置
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = "./data/low_data.txt"
    
    if args.tokenizer_name:
        config.tokenizer.tokenizer_name = args.tokenizer_name
        config.tokenizer.tokenizer_root = f"./data/{args.tokenizer_name}/"
    
    train_tokenizer(data_path, config)