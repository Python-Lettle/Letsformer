from letsformer.util import merge_token, maxInDict
import os
import regex

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
        Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
    """
    # ==================================================
    # Input Data
    # ==================================================
    # 加速遍历，可以不做这个操作
    special_tokens: tuple[str] = tuple(special_tokens)
    file = open(input_path, "r", encoding="utf-8")
    # data: 输入的全部文本数据
    data: str = file.read()
    
    # ----------------------------------------
    # 1 Vocabulary initialization
    # ----------------------------------------
    # 1.1 初始化 0-255 词汇表
    vocab: dict[int, bytes] = {i : bytes([i]) for i in range(256)}
    vocab_idx: int = 256
    vocab_token_set = set[bytes](vocab.values())    # 用于查询是否已经存在 token
    # 1.2 添加 special tokens
    for sp_token in special_tokens:
        if len(vocab) >= vocab_size:  # 如果词汇表已满，停止添加
            break
        sp_token_bytes = sp_token.encode("utf-8")
        if sp_token_bytes not in vocab_token_set:  # 避免重复添加
            vocab[vocab_idx] = sp_token_bytes
            vocab_idx += 1
            vocab_token_set.add(sp_token_bytes)

    # ----------------------------------------
    # 2 Pre-tokenize
    # ----------------------------------------
    token_frequency: dict[tuple[bytes], int] = {}      # token(tuple) 出现频率

    # 2.1 按 special tokens 进行分割
    parts: list[str] = regex.split('|'.join(map(regex.escape,special_tokens)),data)

    # 2.2 利用正则提取单词
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        for word in regex.findall(PAT, part):
            word_bytes: list[bytes] = word.encode("utf-8")
            bytes_list = [bytes([x]) for x in word_bytes]
            # 顺便统计 token 出现频率，保存到 token_frequency
            token_frequency[tuple(bytes_list)] = token_frequency.get(tuple(bytes_list), 0) + 1

    # ----------------------------------------
    # 3 Merge
    # ----------------------------------------
    merges: list[tuple[bytes, bytes]] = []

    # 3.1 初始化 pairs
    # pairs 存储每个 token 的位置 i 和 i+1 组成的 pair 出现频率
    # pairs 用于每一轮的 merge 操作
    pairs: dict[tuple, int] = {}
    # 从 token_frequency 中提取 pairs
    for token in token_frequency.keys():
        for i in range(len(token) - 1):
            new_pair: tuple = (token[i], token[i+1])
            pairs[new_pair] = pairs.get(new_pair, 0) + token_frequency[token]

    # 3.2 开始 merge
    round = 1       # 用于输出调试信息，无实际意义
    while len(vocab) < vocab_size:
        # 判断 pairs 是否为空，为空则停止 merge
        if not pairs:
            break
            
        # 1 找出最大 pair
        max_pair, max_value = maxInDict(pairs)      # max_pair 是 pairs 中 value 最大且字典序最大的 key

        # merge 后的 pair 元素，是一个 bytes (例如 b"ab")
        merged_pair: bytes = max_pair[0] + max_pair[1]
        # 顺便将 merged_pair 添加到 vocab
        vocab[vocab_idx] = merged_pair
        vocab_idx += 1

        # 2 找出所有受影响的 token (即包含 max_pair 的 token)
        affected_tokens = []
        for token, freq in token_frequency.items():
            has_pair = any(token[i:i+2] == max_pair for i in range(len(token) - 1))
            if has_pair:
                affected_tokens.append((token, freq))

        # 3 更新 token_frequency 和 pairs
        for token, freq in affected_tokens:
            # 从 pairs 中删除该 token 的所有 pair
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pairs[pair] -= freq
                if pairs[pair] <= 0:
                    del pairs[pair]
            
            # 将 token 中所有 max_pair 合并为 merged_pair
            new_token = merge_token(token, max_pair, merged_pair)
            
            # 更新 pairs，添加新 token 的所有 pair
            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
            
            # 更新 token_frequency
            del token_frequency[token]
            token_frequency[new_token] = token_frequency.get(new_token, 0) + freq

        # 4 记录本次 merge
        merges.append(max_pair)
        round += 1

    return vocab, merges

