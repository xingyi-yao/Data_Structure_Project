"""B 的数据模块：文本 -> token IDs -> (x, y) batch。

与 A/C 的接口约定：
- C 的 CharTokenizer.encode(text) -> list[int]
- A 的 GPT(idx, targets) -> logits, loss
  - idx: [B, T]，targets: [B, T]
  - targets 已由数据层右移，模型内部不再移位

所以原始 token 序列 [t0, t1, t2, t3, t4] 要切成：
  x = [t0, t1, t2, t3]
  y = [t1, t2, t3, t4]
"""

from pathlib import Path
import torch

from src.tokenizer.char_tokenizer import CharTokenizer


class TextDataset:
    """加载一段文本，建词表，划分 train/val，提供 get_batch。

    用法：
        dataset = TextDataset.from_file("data.txt", context_length=128)
        x, y = dataset.get_batch("train", batch_size=8, device="cuda")
        logits, loss = model(x, y)
    """

    def __init__(self, token_ids, context_length=128, train_ratio=0.9,
                 tokenizer=None):
        """
        Args:
            token_ids: list[int]，已经 encode 好的 token 序列。
            context_length: 一次喂给模型的 token 数（T）。
            train_ratio: train 占比，剩下的做 val。
            tokenizer: 可选，传入已建好的 CharTokenizer；不传则不保存。
        """
        if len(token_ids) < context_length + 2:
            raise ValueError(
                f"token 数 ({len(token_ids)}) 太少，至少需要 context_length+2 "
                f"({context_length + 2}) 才能切出一个 (x, y) 对"
            )

        self.tokenizer = tokenizer
        self.context_length = context_length

        # 按比例划分 train / val。
        split_idx = int(len(token_ids) * train_ratio)
        self.train_ids = token_ids[:split_idx]
        self.val_ids = token_ids[split_idx:]

        # 防止 val 太短切不出一个完整窗口。
        if len(self.val_ids) < context_length + 1:
            self.val_ids = self.train_ids[-(context_length + 1):]

    @classmethod
    def from_text(cls, text, context_length=128, train_ratio=0.9,
                  tokenizer=None):
        """从文本字符串构建数据集。

        如果 tokenizer 为 None，用当前文本新建 CharTokenizer。
        如果传入 tokenizer，则复用已有词表（预训练/微调共用同一份词表时用）。
        """
        if tokenizer is None:
            tokenizer = CharTokenizer(text)
        token_ids = tokenizer.encode(text)
        return cls(token_ids, context_length, train_ratio, tokenizer)

    @classmethod
    def from_file(cls, path, context_length=128, train_ratio=0.9,
                  tokenizer=None, encoding="utf-8"):
        """从文本文件构建数据集。"""
        text = Path(path).read_text(encoding=encoding)
        return cls.from_text(text, context_length, train_ratio, tokenizer)

    def get_batch(self, split, batch_size, device):
        """随机采样一个 batch 的 (x, y)。

        Args:
            split: "train" 或 "val"。
            batch_size:  batch 大小 B。
            device: "cpu" 或 "cuda"。

        Returns:
            x: [B, T] torch.long
            y: [B, T] torch.long（右移一位的标签）
        """
        ids = self.train_ids if split == "train" else self.val_ids
        T = self.context_length

        # 起始位置的合法范围：[0, len(ids) - T - 1]
        # 这样 x = ids[i:i+T]，y = ids[i+1:i+T+1] 都能取到。
        max_start = len(ids) - T - 1
        if max_start <= 0:
            raise ValueError(
                f"{split} 数据太短 ({len(ids)} tokens)，至少需要 {T + 2}"
            )

        # 随机选 batch_size 个起始位置。
        starts = torch.randint(0, max_start, (batch_size,))

        x_list = []
        y_list = []
        for s in starts.tolist():
            x_list.append(ids[s : s + T])
            y_list.append(ids[s + 1 : s + T + 1])

        x = torch.tensor(x_list, dtype=torch.long, device=device)
        y = torch.tensor(y_list, dtype=torch.long, device=device)
        return x, y

    def __len__(self):
        return len(self.train_ids) + len(self.val_ids)


if __name__ == "__main__":
    # 自测：用一小段文本验证 x/y 是否正确右移。
    torch.manual_seed(42)

    text = "hello world, hello world!"
    dataset = TextDataset.from_text(text, context_length=4, train_ratio=0.8)

    print(f"vocab_size: {dataset.tokenizer.vocab_size}")
    print(f"train tokens: {len(dataset.train_ids)}")
    print(f"val tokens: {len(dataset.val_ids)}")

    x, y = dataset.get_batch("train", batch_size=2, device="cpu")
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # 验证右移：y 应该是 x 往后挪一位。
    print(f"x[0]: {x[0].tolist()}")
    print(f"y[0]: {y[0].tolist()}")
    assert torch.equal(y[0, :-1], x[0, 1:]), "y 必须是 x 右移一位"
    print("右移检查通过！")
    print("dataset.py 自测 OK")
