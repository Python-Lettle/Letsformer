
# ==============================
# BPE Tokenizer Utils
# ==============================
def merge_token(
    token_tuple: tuple[bytes],
    best_pair: tuple[bytes,bytes],
    best_token: bytes
) -> tuple[bytes]:
    """将一个 token 中所有的 best_pair 合并为 best_token"""
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