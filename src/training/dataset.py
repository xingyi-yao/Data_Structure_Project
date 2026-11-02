from pathlib import Path
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """连续 token 窗口；标签只在此处右移一位。"""
    def __init__(self, tokens, context_length):
        if type(context_length) is not int or context_length < 1:
            raise ValueError("context_length 必须为正整数")
        self.tokens = torch.as_tensor(tokens, dtype=torch.long).cpu()
        if self.tokens.ndim != 1 or len(self.tokens) <= context_length:
            raise ValueError(f"每个数据集至少需要 {context_length + 1} 个 token")
        if (self.tokens < 0).any():
            raise ValueError("token ID 不能为负数")
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self.tokens[index:index + self.context_length + 1]
        return window[:-1], window[1:]

    def get_batch(self, batch_size, device="cpu", generator=None):
        return get_batch(self, batch_size, device, generator)


def get_batch(dataset, batch_size, device="cpu", generator=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    indices = torch.randint(len(dataset), (batch_size,), generator=generator)
    pairs = [dataset[int(i)] for i in indices]
    x, y = zip(*pairs)
    return torch.stack(x).to(device), torch.stack(y).to(device)


def split_text(text, val_fraction=0.1):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction 必须介于 0 和 1")
    split = int(len(text) * (1 - val_fraction))
    if split == 0 or split == len(text):
        raise ValueError("语料太短，无法拆分训练集与验证集")
    return text[:split], text[split:]


def prepare_data(raw_path, output_dir, val_fraction=0.1, val_path=None):
    text = Path(raw_path).read_text(encoding="utf-8")
    train, val = (text, Path(val_path).read_text(encoding="utf-8")) if val_path else split_text(text, val_fraction)
    if not train or not val:
        raise ValueError("训练文本和验证文本均不能为空")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("train", train), ("val", val)):
        (directory / f"{name}.txt").write_text(content, encoding="utf-8")
    return train, val
