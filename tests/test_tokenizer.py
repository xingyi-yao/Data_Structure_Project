"""运行：python -m unittest discover -s tests（也可用 pytest）。"""

import unittest
import tempfile
from pathlib import Path

from src.tokenizer.char_tokenizer import CharTokenizer, Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer("hello, world!\n")

    def test_round_trip_and_vocab_size(self):
        for text in ("", "llll", "hello", "hello, world!\n", "!\n, "):
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        self.assertEqual(self.tokenizer.vocab_size, len(set("hello, world!\n")))
        self.assertIs(Tokenizer, CharTokenizer)

    def test_stable_ids(self):
        same = CharTokenizer("\n!dlrow ,olleh")
        self.assertEqual(self.tokenizer.encode("hello"), same.encode("hello"))

    def test_unknown_character_and_id(self):
        with self.assertRaisesRegex(ValueError, "不在训练词表"):
            self.tokenizer.encode("z")
        with self.assertRaisesRegex(ValueError, "不在词表"):
            self.tokenizer.decode([self.tokenizer.vocab_size])

    def test_empty_training_text(self):
        with self.assertRaises(ValueError):
            CharTokenizer("")

    def test_save_load_preserves_id_order(self):
        # 特意用非排序词表，检查 load 没有偷偷改变训练时的 ID 顺序。
        tokenizer = CharTokenizer.from_dict({"type": "char", "version": 1,
                                             "chars": ["中", "a", "\n"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = CharTokenizer.load(path)
        self.assertEqual(loaded.encode("a中\n"), [1, 0, 2])
        self.assertEqual(loaded.decode([1, 0, 2]), "a中\n")

    def test_invalid_vocab_and_id_type(self):
        with self.assertRaises(ValueError):
            CharTokenizer.from_dict({"type": "char", "version": 1, "chars": ["a", "a"]})
        with self.assertRaises(TypeError):
            self.tokenizer.decode([1.0])


if __name__ == "__main__":
    unittest.main()
