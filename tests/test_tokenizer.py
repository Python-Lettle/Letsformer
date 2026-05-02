from letsformer import BPETokenizer

DATASET_PATH = "./data/"

def test_tokenizer_encode():
    '''
        测试 tokenizer 是否能够正确编码输入数据
        应当先使用 low_data 数据集产生 vocab 和 merges 文件
    '''
    data_path = DATASET_PATH + "low_data.txt"
    string = ""
    with open(data_path, "r") as f:
        string = f.readline().strip()
    
    tokenizer = BPETokenizer.from_files("data/tokenizer/vocab.pkl", "data/tokenizer/merges.pkl")
    
    tokens = tokenizer.encode(string)

    assert tokens == [108, 111, 119, 32, 108, 111, 119, 32, 108, 111, 119, 32, 108, 111, 119, 32, 108, 111, 119]

def test_tokenizer_decode():
    '''
        测试 tokenizer 是否能够正确解码输入数据
        应当先使用 low_data 数据集产生 vocab 和 merges 文件
    '''
    data_path = DATASET_PATH + "low_data.txt"
    answer = ""
    with open(data_path, "r") as f:
        answer = f.readline().strip()

    tokenizer = BPETokenizer.from_files("data/tokenizer/vocab.pkl", "data/tokenizer/merges.pkl")
    
    tokens = tokenizer.encode(answer)
    text = tokenizer.decode(tokens)
    
    assert text == answer
