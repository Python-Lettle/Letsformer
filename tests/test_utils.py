from letsformer.util import get_pairs, pre_tokenize, merge_token

def test_get_pairs():
    byte_list = [b"a", b"b", b"c", b"d"]
    pairs = get_pairs(byte_list)
    assert pairs == [(b"a", b"b"), (b"b", b"c"), (b"c", b"d")]

def test_pre_tokenize():
    '''
        测试 pre_tokenize 函数是否能够正确处理输入数据
    '''
    data = "hello world<|endoftext|>"
    tokens = pre_tokenize(data, special_tokens=["<|endoftext|>"])
    assert tokens == [b'hello', b' world', b'<|endoftext|>']

def test_merge_token():
    '''
        测试 merge_token 函数是否能够正确合并 token
    '''
    token_tuple = (b'a', b'b', b'c', b'd')
    best_pair = (b'b', b'c')
    best_token = b'egg'
    merged_token = merge_token(token_tuple, best_pair, best_token)
    assert merged_token == (b'a', b'egg', b'd')