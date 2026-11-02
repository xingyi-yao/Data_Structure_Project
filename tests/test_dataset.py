"""数据边界、标签位移和可重复采样；按用户要求，本次未执行。"""
import unittest
import torch
from src.training.dataset import TextDataset, split_text


class TestDataset(unittest.TestCase):
    def test_last_window_and_shift(self):
        dataset = TextDataset([0, 1, 2, 3, 4], 3)
        self.assertEqual(len(dataset), 2)
        x, y = dataset[1]
        self.assertEqual(x.tolist(), [1, 2, 3])
        self.assertEqual(y.tolist(), [2, 3, 4])

    def test_minimum_and_invalid_length(self):
        self.assertEqual(len(TextDataset([0, 1], 1)), 1)
        with self.assertRaises(ValueError):
            TextDataset([0, 1], 2)
        with self.assertRaises(ValueError):
            TextDataset([0, 1], 0)

    def test_seeded_sampling(self):
        dataset = TextDataset(list(range(20)), 4)
        first = dataset.get_batch(3, generator=torch.Generator().manual_seed(12))
        second = dataset.get_batch(3, generator=torch.Generator().manual_seed(12))
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[0] + 1, first[1])

    def test_split_preserves_text(self):
        train, val = split_text("abcdefghij", 0.2)
        self.assertEqual(train, "abcdefgh")
        self.assertEqual(val, "ij")


if __name__ == "__main__":
    unittest.main()
