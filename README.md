# Letsformer

## 前言

该项目展示了一个最简单的 Transformer 代码，其架构参考论文《Attention is All You Need》以及 cs336 中的架构。

该项目会将小型数据集一并放入目录 `./data` 中，方便直接使用测试。

## 安装

### 前置要求

- Python 3.13 或更高版本（较低版本请自行试验是否可运行）
- uv 包管理器

### 安装步骤

1. 克隆项目：
   ```bash
   git clone https://github.com/Python-Lettle/Letsformer.git
   cd letsformer
   ```

2. 使用 uv 安装依赖：
   ```bash
   uv sync
   ```

## 使用示例

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_train_bpe.py

# 详细输出模式
uv run pytest -v
```

### 测试用例

```bash
# 1. BPE 训练测试
uv run pytest tests/test_train_bpe.py

# 2. Tokenizer encode decode 测试
# 该测试应当建立在 1. BPE 训练测试 结束后产生的 vocab 和 merges 的基础上进行
uv run pytest tests/test_tokenizer.py
```



## API 参考

### train_bpe()

在给定语料库上训练 BPE 分词器。

**参数：**

- `input_path` (str | os.PathLike): 训练数据文件路径
- `vocab_size` (int): 词表总大小（包括特殊 Token）
- `special_tokens` (list[str]): 特殊 Token 列表

**返回值：**

- `vocab` (dict[int, bytes]): 训练得到的词表，映射 token ID 到 token bytes
- `merges` (list[tuple[bytes, bytes]]): BPE 合并规则列表



## 许可证

本项目使用 MIT 协议，如有问题请联系作者邮箱：1071445082@qq.com
