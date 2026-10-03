"""推理与采样模块。"""

from .generate import generate
from .sampling import sample_next_token

__all__ = ["generate", "sample_next_token"]
