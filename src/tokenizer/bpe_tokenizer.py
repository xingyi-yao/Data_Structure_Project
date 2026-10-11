"""可接入 small_gpt 的字符级 BPE；只依赖 Python 标准库。

接口与 CharTokenizer 一致：encode / decode / vocab_size / save / load /
to_dict / from_dict。encode 返回 list[int]，模型仍接收 torch.long 的 [B,T]。

B 同学的用法（只用训练集学习词表，之后固定词表）：
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    tokenizer = BPETokenizer(train_text, vocab_size=4096)
    ids = tokenizer.encode(train_text)
    tokenizer.save("checkpoints/tokenizer.json")
    config["vocab_size"] = tokenizer.vocab_size  # 使用实际大小，不是目标 4096！
    # checkpoint["tokenizer"] = tokenizer.to_dict()  # 建议随权重保存

C 同学直接使用 BPETokenizer.load(path)，或包内通用 load_tokenizer(path)。
训练词表的命令（在 E:/small_gpt 根目录运行）：
    python src/tokenizer/bpe_tokenizer.py --train-file data/train.txt --vocab-size 4096 --output checkpoints/tokenizer.json

这里从 Unicode 字符开始合并，不是 256 字节版本：中文、英文、空格、换行
均原样保留；不添加 BOS/EOS/PAD，与已有固定长度生成约定一致。
从未在训练集中出现的字符会明确报错，与 CharTokenizer 的行为一致。
更换分词器后 token ID 的含义改变，必须使用对应 BPE 词表训练的模型权重。

教学实现会逐轮统计相邻对；用重复片段的频次减少工作，适合课程规模语料。
大规模语料应使用专业 tokenizer 训练工具；不要为验证集重新训练词表。
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


# 空白、单词、标点分组；findall 不删除任何字符，不做小写化或 strip。
# 同组内允许合并，避免把单词和它旁边的空白整个粘成一个 token。
_PIECES = re.compile(r"\s+|\w+|[^\w\s]+", re.UNICODE)
_PRE_TOKENIZER = "unicode_groups_v1"


def _merge_pair(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """从左向右合并不重叠的相邻对；例如 aaaa 合并 aa 得到 [aa,aa]。"""
    result = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(ids[index])
            index += 1
    return tuple(result)


class BPETokenizer:
    def __init__(self, text: str, vocab_size: int = 4096, min_frequency: int = 2):
        """学习词表；vocab_size 是总 token 数上限，min_frequency 为最低对频次。"""
        if not isinstance(text, str):
            raise TypeError("训练文本必须是 str")
        if not text:
            raise ValueError("训练文本不能为空")
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocab_size 必须为正整数")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency 必须为正整数")
        chars = sorted(set(text))
        if vocab_size < len(chars):
            raise ValueError(f"目标 vocab_size 至少为 {len(chars)}，才能容纳全部基础字符")
        char_ids = {char: index for index, char in enumerate(chars)}

        # Counter 汇总相同片段；同一个单词出现 100 次，仅存一份序列和频次 100。
        sequences = Counter(tuple(char_ids[c] for c in piece) for piece in _PIECES.findall(text))
        merges = []
        while len(chars) + len(merges) < vocab_size:
            counts = Counter()
            for ids, frequency in sequences.items():
                for pair in zip(ids, ids[1:]):
                    counts[pair] += frequency
            if not counts:
                break  # 每个片段已不可再合并。
            # 优先最高频；并列时按 ID 对排序，保证训练可复现。
            pair = min(counts, key=lambda item: (-counts[item], item))
            if counts[pair] < min_frequency:
                break
            new_id = len(chars) + len(merges)
            updated = Counter()
            for ids, frequency in sequences.items():
                updated[_merge_pair(ids, pair, new_id)] += frequency
            sequences = updated
            merges.append(pair)
        self._initialize(chars, merges)

    def _initialize(self, chars, merges):
        """训练/加载共用：ID 由基础字符顺序和合并顺序唯一确定。"""
        self.chars = tuple(chars)  # 仅基础字符；完整词表是 tokens。
        self.merges = tuple(tuple(pair) for pair in merges)
        self._char_ids = {char: index for index, char in enumerate(self.chars)}
        tokens = list(self.chars)
        self._merge_ids = {}
        for pair in self.merges:
            self._merge_ids[pair] = len(tokens)
            tokens.append(tokens[pair[0]] + tokens[pair[1]])
        self.tokens = tuple(tokens)

    @property
    def vocab_size(self) -> int:
        """A 的 Embedding/输出层应使用这个实际词表大小。"""
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """文本 -> ID；只使用学好的规则，不在推理时增加 token。"""
        if not isinstance(text, str):
            raise TypeError("待编码内容必须是 str")
        output = []
        memo = {}  # 本次调用中重复的片段只编码一次。
        for piece in _PIECES.findall(text):
            if piece not in memo:
                try:
                    ids = tuple(self._char_ids[char] for char in piece)
                except KeyError as exc:
                    raise ValueError(f"字符 {exc.args[0]!r} 不在训练词表中") from exc
                while len(ids) > 1:
                    candidates = (pair for pair in zip(ids, ids[1:]) if pair in self._merge_ids)
                    # 新 ID 越小，规则学得越早。编码按训练顺序，而非当前频次或最长匹配。
                    pair = min(candidates, key=self._merge_ids.__getitem__, default=None)
                    if pair is None:
                        break
                    ids = _merge_pair(ids, pair, self._merge_ids[pair])
                memo[piece] = ids
            output.extend(memo[piece])
        return output

    def decode(self, ids: list[int]) -> str:
        """ID -> 文本；每个 token 都是完整字符串，可直接接入现有 UI。"""
        if not isinstance(ids, (list, tuple)) or any(type(index) is not int for index in ids):
            raise TypeError("ids 必须是整数列表；Tensor 请先 .tolist()")
        if any(index < 0 or index >= self.vocab_size for index in ids):
            raise ValueError("token ID 不在词表中")
        return "".join(self.tokens[index] for index in ids)

    def to_dict(self) -> dict:
        """JSON 与 torch.save 均可保存；不仅存字符，还保存有序合并规则。"""
        return {"type": "bpe", "version": 1, "pre_tokenizer": _PRE_TOKENIZER,
                "chars": list(self.chars), "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        """只恢复词表，不重新训练；逐条验证 ID 引用，防止损坏文件被误读。"""
        if (not isinstance(data, dict) or data.get("type") != "bpe"
                or type(data.get("version")) is not int or data["version"] != 1
                or data.get("pre_tokenizer") != _PRE_TOKENIZER):
            raise ValueError("不支持的 BPE 词表类型、版本或预切分规则")
        chars, merges = data.get("chars"), data.get("merges")
        if (not isinstance(chars, list) or not chars
                or any(not isinstance(char, str) or len(char) != 1 for char in chars)
                or len(set(chars)) != len(chars)):
            raise ValueError("chars 必须是不重复的单字符列表，顺序即基础 ID")
        if not isinstance(merges, list):
            raise ValueError("merges 必须是按训练顺序保存的列表")
        seen = set()
        for rank, pair in enumerate(merges):
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(index) is not int or not 0 <= index < len(chars) + rank for index in pair)):
                raise ValueError("每条 merge 必须引用此前已经存在的两个 token ID")
            if tuple(pair) in seen:
                raise ValueError("不允许重复的 merge 规则")
            seen.add(tuple(pair))
        tokenizer = cls.__new__(cls)
        tokenizer._initialize(chars, merges)
        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


Tokenizer = BPETokenizer  # 支持在本模块中使用通用名称；包级旧别名仍保持 CharTokenizer。


def main():
    parser = argparse.ArgumentParser(description="用训练集学习字符级 BPE 词表")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tokenizer.json"))
    args = parser.parse_args()
    # newline='' 保留 Windows 的 CRLF，不让读文件步骤偷偷改变原文。
    with args.train_file.open(encoding="utf-8", newline="") as file:
        text = file.read()
    tokenizer = BPETokenizer(text, args.vocab_size, args.min_frequency)
    tokenizer.save(args.output)
    print(f"基础字符数：{len(tokenizer.chars)}；合并次数：{len(tokenizer.merges)}")
    print(f"实际 vocab_size：{tokenizer.vocab_size}；已保存到：{args.output}")


if __name__ == "__main__":
    main()
