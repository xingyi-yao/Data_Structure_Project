"""B 的 dataset 单元测试。

运行：python -m unittest tests.test_dataset
或：python -m unittest discover -s tests
"""

import unittest
import tempfile
from pathlib import Path

import torch

from src.training.dataset import TextDataset


class TestTextDataset(unittest.TestCase):
    def setUp(self):
        self.text = "hello world, hello world!"
        self.context_length = 4

    def test_from_text_creates_dataset(self):
        dataset = TextDataset.from_text(
            self.text, context_length=self.context_length, train_ratio=0.8)
        self.assertIsNotNone(dataset.tokenizer)
        self.assertEqual(dataset.tokenizer.vocab_size, len(set(self.text)))
        self.assertGreater(len(dataset.train_ids), 0)
        self.assertGreater(len(dataset.val_ids), 0)

    def test_xy_is_shifted_by_one(self):
        """y 必须是 x 右移一位。"""
        dataset = TextDataset.from_text(
            self.text, context_length=self.context_length, train_ratio=0.8)
        x, y = dataset.get_batch("train", batch_size=4, device="cpu")
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_batch_shapes(self):
        """get_batch 返回 [B, T] 的 x 和 y。"""
        dataset = TextDataset.from_text(
            self.text, context_length=self.context_length, train_ratio=0.8)
        batch_size = 3
        x, y = dataset.get_batch("train", batch_size, device="cpu")
        self.assertEqual(x.shape, (batch_size, self.context_length))
        self.assertEqual(y.shape, (batch_size, self.context_length))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_train_val_split(self):
        """train 和 val 的 token 总数等于原始 token 数。"""
        # 用小的 context_length，避免 val 被替换成 train 尾部
        dataset = TextDataset.from_text(
            self.text, context_length=2, train_ratio=0.8)
        total = len(dataset.train_ids) + len(dataset.val_ids)
        self.assertEqual(total, len(dataset.tokenizer.encode(self.text)))

    def test_too_short_text_raises(self):
        """文本太短（不够 context_length+2）应该报错。"""
        with self.assertRaises(ValueError):
            TextDataset.from_text("ab", context_length=10)

    def test_from_file(self):
        """从文件加载应该和 from_text 结果一致。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text(self.text, encoding="utf-8")
            dataset = TextDataset.from_file(
                str(path), context_length=self.context_length, train_ratio=0.8)
        self.assertEqual(dataset.tokenizer.vocab_size, len(set(self.text)))
        x, y = dataset.get_batch("train", batch_size=2, device="cpu")
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_reuse_tokenizer(self):
        """传入已有 tokenizer 时，应该复用而不是新建。"""
        from src.tokenizer.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer(self.text)
        dataset = TextDataset.from_text(
            self.text, context_length=self.context_length,
            train_ratio=0.8, tokenizer=tokenizer)
        self.assertIs(dataset.tokenizer, tokenizer)

    def test_val_too_short_falls_back_to_train_tail(self):
        """val 太短时，应该从 train 尾部取数据补充。"""
        dataset = TextDataset.from_text(
            self.text, context_length=self.context_length, train_ratio=0.95)
        self.assertGreaterEqual(
            len(dataset.val_ids), self.context_length + 2)


if __name__ == "__main__":
    unittest.main()
