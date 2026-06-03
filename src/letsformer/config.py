from dataclasses import dataclass, field
from typing import Optional, List
import json
import os

@dataclass
class TokenizerConfig:
    """Tokenizer 配置"""
    vocab_size: int = 10000
    special_tokens: List[str] = None
    tokenizer_name: str = "tokenizer"
    tokenizer_root: str = "./data/tokenizer/"
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["<|endoftext|>"]
    
    @property
    def vocab_path(self) -> str:
        return os.path.join(self.tokenizer_root, "vocab.pkl")
    
    @property
    def merges_path(self) -> str:
        return os.path.join(self.tokenizer_root, "merges.pkl")

@dataclass
class ModelConfig:
    """模型架构配置"""
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 6000
    batch_size: int = 32
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.001
    valid_interval: int = 150
    gradient_clip: float = 1.0
    print_interval: int = 20

@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_iters: int = 500
    cosine_cycle_iters: int = 10000

@dataclass
class DataConfig:
    """数据配置"""
    train_data_path: str = "./data/tinystories_train.npy"
    valid_data_path: str = "./data/tinystories_valid.npy"
    model_weights_path: Optional[str] = "./data/model/model_base.pt"
    save_model_dir: str = "./data/model/"

@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "cuda"
    seed: int = 42

@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 5000
    temperature: float = 0.8
    top_k: int = 5
    eos_token: str = "<|endoftext|>"
    device: str = "cuda"

@dataclass
class TrainConfig:
    """完整训练配置"""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainConfig":
        """从字典创建配置"""
        return cls(
            tokenizer=TokenizerConfig(**config_dict.get("tokenizer", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
            data=DataConfig(**config_dict.get("data", {})),
            system=SystemConfig(**config_dict.get("system", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> "TrainConfig":
        """从 JSON 文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "tokenizer": self.tokenizer.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "scheduler": self.scheduler.__dict__,
            "data": self.data.__dict__,
            "system": self.system.__dict__,
            "generation": self.generation.__dict__,
        }
    
    def to_json(self, json_path: str):
        """保存到 JSON 文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

DEFAULT_CONFIG = TrainConfig()