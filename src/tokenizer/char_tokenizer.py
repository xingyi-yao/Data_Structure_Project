"""字符级 Tokenizer；B 训练时保存词表，C 推理时按同一 ID 顺序加载。"""

import json
from pathlib import Path


class CharTokenizer:
    """词表一旦建立就固定；训练与推理必须使用同一份词表。"""

    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError("训练文本必须是 str")
        if not text:
            raise ValueError("训练文本不能为空：空词表无法用于模型")

        # set 去重，sorted 固定 ID 顺序：相同训练文本总得到相同词表。
        self.chars = tuple(sorted(set(text)))
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for char, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        """将字符串变成 token ID 列表，供 B 的训练和 C 的推理使用。"""
        if not isinstance(text, str):
            raise TypeError("待编码内容必须是 str")
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            # 不临时扩充词表，否则 A 的模型输出维度和已有 ID 会失效。
            raise ValueError(f"字符 {exc.args[0]!r} 不在训练词表中") from exc

    def decode(self, ids: list[int]) -> str:
        """把 token ID 逐个查回字符，再连接成原文。"""
        if any(type(token_id) is not int for token_id in ids):
            raise TypeError("ids 必须是 Python 整数列表；Tensor 请先 .tolist()")
        try:
            return "".join(self.itos[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"token ID {exc.args[0]!r} 不在词表中") from exc

    @property
    def vocab_size(self) -> int:
        """词表大小 V；A 创建 Embedding 和模型输出层时使用。"""
        return len(self.chars)


    def to_dict(self) -> dict:
        """chars[0] 的 ID 为 0；恢复时不可重新排序。"""
        return {"type": "char", "version": 1, "chars": list(self.chars)}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        if not isinstance(data, dict) or data.get("type") != "char" or data.get("version") != 1:
            raise ValueError("需要 type='char', version=1 的字符词表")
        chars = data.get("chars")
        if (not isinstance(chars, list) or not chars
                or any(not isinstance(c, str) or len(c) != 1 for c in chars)
                or len(set(chars)) != len(chars)):
            raise ValueError("chars 必须是非空、不重复的单字符列表，顺序对应 ID")
        tokenizer = cls.__new__(cls)
        tokenizer.chars = tuple(chars)
        tokenizer.stoi = {char: i for i, char in enumerate(chars)}
        tokenizer.itos = dict(enumerate(chars))
        return tokenizer

    def save(self, path: str | Path) -> None:
        """TODO(B)：训练建立词表后调用 tokenizer.save('checkpoints/tokenizer.json')。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# 工作流程同时使用 Tokenizer 与 CharTokenizer 两种称呼；保持同一个实现。
Tokenizer = CharTokenizer
