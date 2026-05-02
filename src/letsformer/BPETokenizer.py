from letsformer.util import merge_token, maxInDict, pre_tokenize
import os
import regex
import pickle
from collections.abc import Iterable, Iterator

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

class TokenIterator(Iterator[int]):
    def __init__(self, tokenizer, iterable: Iterable[str]):
        # 传进来的可迭代数据对象
        self.iterable: Iterable[str] = iterable
        self.current_tokens: list[int] = []
        self.current_idx: int = 0

        self.tokenizer = tokenizer

    def __iter__(self):
        return self
    def __next__(self) -> int:
        if self.current_idx >= len(self.current_tokens):
            # 当前读取的 current_tokens 已经全部返回完毕, 应该读取并解析新的 token
            self.get_new_tokens()
        # 此时应当有可返回的新 token bytes (int)
        result = self.current_tokens[self.current_idx]
        self.current_idx += 1
        return result

    def get_new_tokens(self):
        # 先取出 iterable 的一个行, 初始化 token 列表
        current_line = next(self.iterable)
        self.current_tokens = self.tokenizer.encode(current_line)
        self.current_idx = 0

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        '''
        利用给定的 vocabulary、merges 和一组(可选的) special tokens 构造 tokenizer

        函数接收以下参数:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        '''
        self.vocab_int_bytes: dict[int, bytes] = vocab
        # 构建一个反向 dict
        self.vocab_bytes_int: dict[bytes, int] = {v : k for k, v in vocab.items()}

        self.merges: list[tuple[bytes, bytes]] = merges
        self.merges: tuple[tuple[bytes, bytes]] = tuple(self.merges)    # tuple 加速遍历

        # 构建一个快表
        self.merges_dict: dict[tuple[bytes, bytes], bytes] = {}
        for merge in self.merges:
            self.merges_dict[merge] = merge[0] + merge[1]

        self.special_tokens: list[str] = special_tokens if special_tokens else []  # 确保 special_tokens 是一个列表
        # 给 special tokens 按长度排序
        self.special_tokens.sort(key=lambda x: len(x), reverse=True)
        self.special_tokens_bytes: list[bytes] = [x.encode("utf-8") for x in self.special_tokens]

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        '''
        一个类方法，用于从序列化的 vocabulary 和 merges（格式与BPE training 代码输出相同）以及(可选的) special tokens 中构建并返回一个 Tokenizer

        函数接收以下参数:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        '''
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        '''
        将输入文本编码为一系列 token IDs
        '''
        # 把输入的字符串分割成 pre-token, 例如: [b'the', b' cat', b' ate']
        pre_tokens = pre_tokenize(text, special_tokens=self.special_tokens)
        
        # 遍历处理每一个 pre-token, 得到 encoded tokens
        encoded_tokens: list[int] = []
        for token_bytes in pre_tokens:      # 遍历每个 token (bytes)
            # 如果是 special token
            if token_bytes in self.special_tokens_bytes:
                encoded_tokens.append(self.vocab_bytes_int[token_bytes])
                continue
            
            # 不是 special token, 准备进行 merge
            token_bytes_list = [bytes([x]) for x in token_bytes]    # 将一个 token 拆成原始字符, 例如: b'egg' -> [b'e', b'g', b'g']
            flag = True
            while flag:
                flag = False        # 如果本次循环没有进行 merge 操作, 即本次循环已经完成了全部 merge
                for merge in self.merges:   # 当前检查的 merge
                    i = 0
                    while i < len(token_bytes_list) - 1:
                        if token_bytes_list[i] == merge[0] and token_bytes_list[i+1] == merge[1]:
                            # 在 token 中找到了当前的 merge, 对 token 进行修改
                            token_bytes_list[i] = self.merges_dict[merge]
                            del token_bytes_list[i+1]
                            flag = True
                        else:
                            # 位置 i 不是该 merge 项
                            i += 1

            # 将 merge 后的 token 利用 vocab 转化为编码
            for merged in token_bytes_list:
                encoded_tokens.append(self.vocab_bytes_int[merged])

        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        给定一个字符串的可迭代对象(例如，一个 Python 文件句柄)，返回一个生成器
        该生成器以惰性方式生成 token IDs
        这对于我们无法直接加载到内存中的大型文件的 memory-efficient tokenization 是必需的
        '''
        token_it_class = TokenIterator(self, iterable)
        token_it = iter(token_it_class)
        return token_it

    def decode(self, ids: list[int]) -> str:
        '''
        将一系列 token IDs 解码为文本
        '''
        text: bytes = b""
        for id in ids:
            text += self.vocab_int_bytes[id]

        return str(text, encoding="utf-8", errors='replace')
