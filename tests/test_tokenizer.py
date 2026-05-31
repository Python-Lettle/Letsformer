from letsformer import BPETokenizer
from letsformer.debug import console

DATASET_PATH = "./data/"

def test_tokenizer_encode():
    '''
        测试 tokenizer 是否能够正确编码和解码输入数据
        应当先使用 low_data 数据集产生 vocab 和 merges 文件
    '''
    data_path = DATASET_PATH + "low_data.txt"
    string = ""
    with open(data_path, "r") as f:
        string = f.read().strip()
    
    tokenizer = BPETokenizer.from_files("data/tokenizer/vocab.pkl", "data/tokenizer/merges.pkl", special_tokens=["<|endoftext|>"])
    
    tokens = tokenizer.encode(string)
    text = tokenizer.decode(tokens)
    assert text == string

