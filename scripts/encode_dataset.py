'''
    对一个文本文件使用给定的 Tokenizer 进行编码, 将编码后的 token 数组保存到文件中

    调用方式: uv run scripts/encode_dataset.py <data_path> <output_path>

    参数:
        sys.argv[1]: str, 训练集路径
        sys.argv[2]: str, 输出文件路径
'''
from letsformer import BPETokenizer
from letsformer.debug import console, DEBUG
import numpy as np
from tqdm import tqdm
import sys
import os

TOKENIZER_NAME = "tinystories_tokenizer"
VOCAB_PATH = "./data/" + TOKENIZER_NAME + "/vocab.pkl"
MERGES_PATH = "./data/"+ TOKENIZER_NAME + "/merges.pkl"
SPECIAL_TOKENS = ["<|endoftext|>"]

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
    
    # 2. 统计 token 数
    total_tokens = 0
    tokens = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            token = tokenizer.encode(line)
            tokens.extend(token)
            total_tokens += len(token)
    console.print("total_tokens:", total_tokens)
    
    # 3. 创建保存的 memmap 文件
    tokens_mm = np.memmap(output_path, dtype=np.int32, mode='w+', shape=(total_tokens,))

    # 4. 写入
    tokens_mm[:total_tokens] = np.array(tokens, dtype=np.int32)
    tokens_mm.flush()

def encode_dataset(data_path, output_path):
    # 1. Load tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=SPECIAL_TOKENS,
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
    encode_dataset(sys.argv[1], sys.argv[2])
    
