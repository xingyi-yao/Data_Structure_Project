"""统一词表加载入口：根据 JSON 的 type 自动选择 CharTokenizer 或 BPETokenizer。"""

import json
from pathlib import Path

from .char_tokenizer import CharTokenizer, Tokenizer
from .bpe_tokenizer import BPETokenizer

def tokenizer_from_dict(data: dict) -> CharTokenizer | BPETokenizer:
    if not isinstance(data, dict):
        raise TypeError("tokenizer 配置必须是字典")
    kind = data.get("type")
    if kind == "char":
        return CharTokenizer.from_dict(data)
    if kind == "bpe":
        return BPETokenizer.from_dict(data)
    raise ValueError(f"不支持的 tokenizer 类型：{kind!r}")


def load_tokenizer(path: str | Path) -> CharTokenizer | BPETokenizer:
    return tokenizer_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# 不改变旧代码中 Tokenizer=CharTokenizer 的含义；训练 BPE 时显式用 BPETokenizer。
__all__ = ["CharTokenizer", "BPETokenizer", "Tokenizer", "load_tokenizer", "tokenizer_from_dict"]
