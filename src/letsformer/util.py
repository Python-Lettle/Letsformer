import regex
from letsformer.debug import console, DEBUG
# ==============================
# BPE Tokenizer Utils
# ==============================
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_token(
    token_tuple: tuple[bytes],
    best_pair: tuple[bytes,bytes],
    best_token: bytes
) -> tuple[bytes]:
    """将一个 token tuple 中所有的 best_pair 合并为 best_token"""
    merged_token = []
    i = 0
    while i < len(token_tuple):
        # 检查位置 i 是否有 best_pair
        if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == best_pair:
            # 发现 best_pair，将合并后的 best_token 加入 merged_token
            merged_token.append(best_token)
            i += 2
        else:
            # 该位置没有 best_pair，将该位置添加到 merged_token
            merged_token.append(token_tuple[i])
            i += 1
    return tuple(merged_token)

def maxInDict(d: dict) -> tuple[tuple[bytes], int]:
    '''
        寻找 dict 中 value 最大 且 key 的字典序最大的 key
        用于从 pairs 中筛选本次 merge 的 pair
    '''
    max_value: int = 0
    max_pair : tuple = ()
    for k,v in d.items():
        if v > max_value:
            # 找到频率更高的 pair
            max_value = v
            max_pair = k
        elif v == max_value:
            # 选择字典序最大的那个 pair
            max_pair = max(max_pair, k)
    return max_pair, max_value

def get_pairs(byte_list: list[bytes]) -> list[tuple[bytes,bytes]]:
    '''
        从 byte_list 中提取所有可能的 pair
    '''
    pairs: list[tuple[bytes,bytes]] = [(byte_list[i], byte_list[i+1])  for i in range(len(byte_list)-1)]
    return pairs

def pre_tokenize(data: str, special_tokens: list[str] = []) -> list[bytes]:
    '''
        预处理文本，将 special tokens 和单词提取出来
        并将单词转换为 bytes 类型
        最后返回一个 bytes 列表
        用于后续的 tokenization

        参数:
            data: 输入的文本字符串
            special_tokens: 特殊 tokens 列表，默认为空
        返回:
            bytes 列表，包含所有单词的 bytes 表示
    '''
    # 1 处理 data
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(map(regex.escape, sorted_special_tokens))
    if pattern:
        parts = regex.split(f"({pattern})", data)
    else:
        parts = [data]

    # 2 利用正则提取单词
    result: list[bytes] = []
    
    for part in parts:
        if part in special_tokens:
            # special tokens 直接处理
            result.append(part.encode("utf-8"))
            continue
        for word in regex.findall(PAT, part):
            word_bytes: bytes = word.encode("utf-8")
            result.append(word_bytes)
            # bytes_list = [bytes([x]) for x in word_bytes]

    return result