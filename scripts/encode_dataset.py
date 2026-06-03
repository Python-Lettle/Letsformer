'''
    对一个文本文件使用给定的 Tokenizer 进行编码, 将编码后的 token 数组保存到文件中

    调用方式: uv run scripts/encode_dataset.py [data_path] [output_path] [--config CONFIG_PATH]

    参数:
        data_path: str, 训练集路径
        output_path: str, 输出文件路径
'''
from letsformer import BPETokenizer
from letsformer.debug import console, DEBUG
from letsformer.config import TrainConfig, DEFAULT_CONFIG
import numpy as np
from tqdm import tqdm
import sys
import os
import argparse

def encode_txt_nparray(tokenizer: BPETokenizer, input_path: str | os.PathLike, output_path: str | os.PathLike):
    '''
    对一个文本文件使用给定的 Tokenizer 进行编码, 将编码后的 token 数组保存到文件中
    
    Args:
        tokenizer: BPETokenizer, 用于编码的 BPE 令牌化器
        input_path: str, 训练集路径
        output_path: str, 输出文件路径
    '''
    # 1. 读取文件, 统计行数
    with open(input_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    console.print("num_lines:", num_lines)
    
    # 2. 编码
    total_tokens = 0
    tokens = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Encoding lines"):
            token = tokenizer.encode(line)
            tokens.extend(token)
            total_tokens += len(token)
    console.print("total_tokens:", total_tokens)
    
    # 3. 创建保存的 memmap 文件
    tokens_mm = np.memmap(output_path, dtype=np.int32, mode='w+', shape=(total_tokens,))

    # 4. 写入
    tokens_mm[:total_tokens] = np.array(tokens, dtype=np.int32)
    tokens_mm.flush()

def encode_dataset(data_path, output_path, config: TrainConfig = DEFAULT_CONFIG):
    # 1. Load tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=config.tokenizer.vocab_path,
        merges_filepath=config.tokenizer.merges_path,
        special_tokens=config.tokenizer.special_tokens,
    )
    if DEBUG: console.print(f"(encode_dataset) Tokenizer loaded, vocab size: {len(tokenizer.vocab_bytes_int)}, merges size: {len(tokenizer.merges)}")

    # 2. Encode dataset
    encode_txt_nparray(
        tokenizer = tokenizer,
        input_path= data_path,
        output_path=output_path,
    )

    if DEBUG: console.print(f"(encode_dataset) Dataset encoded, output path: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode Dataset with Tokenizer")
    parser.add_argument("data_path", nargs="?", help="Path to input text data")
    parser.add_argument("output_path", nargs="?", help="Path to output encoded file")
    parser.add_argument("--config", help="Path to config JSON file")
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = TrainConfig.from_json(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # 如果命令行指定了参数，覆盖配置
    data_path = args.data_path if args.data_path else "./data/low_data.txt"
    output_path = args.output_path if args.output_path else "./data/train.bin"
    
    encode_dataset(data_path, output_path, config)
    
