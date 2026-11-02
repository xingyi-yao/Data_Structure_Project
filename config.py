"""项目默认配置；路径不依赖启动时的工作目录。"""
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    n_embd: int = 64
    n_layer: int = 2
    dropout: float = 0.1
    n_head: int = 4

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    eval_interval: int = 100
    eval_batches: int = 20
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"


def normalize_model_config(config):
    """兼容已有测试/旧 checkpoint 的长名称；拒绝冲突或未知字段。"""
    result = dict(config)
    for old, new in {"embedding_dim": "n_embd", "num_layers": "n_layer",
                     "num_heads": "n_head", "block_size": "context_length"}.items():
        if old in result:
            value = result.pop(old)
            if new in result and result[new] != value:
                raise ValueError(f"配置冲突：{old} / {new}")
            result[new] = value
    # 旧 GPT 构造函数在未提供 n_head 时使用单头，恢复时保留该语义。
    result.setdefault("n_head", 1)
    return ModelConfig(**result).to_dict()
