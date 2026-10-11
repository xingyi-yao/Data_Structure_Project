"""标准库测试：python -m unittest tests.test_bpe_tokenizer -v

也可运行 python -m pytest tests -q。
安装 PyTorch 后，额外验证 BPE checkpoint -> ModelService -> generate。
"""

import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

from src.tokenizer import CharTokenizer, BPETokenizer, load_tokenizer, tokenizer_from_dict


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "hello hello, world!\r\n学中文 学中文。\t123_123 👋 e\u0301\x00"
        self.tokenizer = BPETokenizer(self.text, vocab_size=100)

    def test_exact_round_trip(self):
        for text in ("", self.text, "hello", "学中文", "文中学", " \t\r\n", "👋", "e\u0301\x00"):
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(ids), text)
                self.assertTrue(all(type(i) is int and 0 <= i < self.tokenizer.vocab_size for i in ids))

    def test_frequent_pairs_reduce_length(self):
        tokenizer = BPETokenizer("aaaa aaaa", vocab_size=20)
        self.assertLess(len(tokenizer.encode("aaaa aaaa")), len("aaaa aaaa"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("aaaa aaaa")), "aaaa aaaa")
        self.assertLess(tokenizer.vocab_size, 20)  # 达不到目标上限时应提前停止。

    def test_merge_rank_not_longest_match(self):
        tokenizer = BPETokenizer.from_dict({"type": "bpe", "version": 1,
            "pre_tokenizer": "unicode_groups_v1", "chars": ["a", "b", "c"],
            "merges": [[1, 2], [0, 1]]})
        self.assertEqual(tokenizer.encode("abc"), [0, 3])  # bc 比 ab 更早学到。
        self.assertEqual(tokenizer.decode([0, 3]), "abc")

    def test_nonoverlap_and_determinism(self):
        first = BPETokenizer("aaaa", vocab_size=2)
        self.assertEqual(first.tokens, ("a", "aa"))
        self.assertEqual(first.encode("aaaaa"), [1, 1, 0])
        second = BPETokenizer("aaaa", vocab_size=2)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_whitespace_boundaries(self):
        tokenizer = BPETokenizer("ab ab\n ab ab", vocab_size=100, min_frequency=1)
        # 学习的单词 token 不能吞掉它两侧的空白。
        self.assertTrue(all(token.isspace() or not any(c.isspace() for c in token)
                            for token in tokenizer.tokens))

    def test_limit_equal_to_alphabet(self):
        tokenizer = BPETokenizer("ab ab", vocab_size=3)
        self.assertEqual(tokenizer.merges, ())
        self.assertEqual(len(tokenizer.encode("ab ab")), 5)

    def test_no_merge_below_min_frequency(self):
        tokenizer = BPETokenizer("abab", vocab_size=10, min_frequency=3)
        self.assertEqual(tokenizer.vocab_size, 2)

    def test_unknown_character_and_invalid_ids(self):
        with self.assertRaisesRegex(ValueError, "不在训练词表"):
            self.tokenizer.encode("新")
        for ids in ([-1], [self.tokenizer.vocab_size]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.tokenizer.decode(ids)
        for ids in ([True], [1.0], "12"):
            with self.subTest(ids=ids), self.assertRaises(TypeError):
                self.tokenizer.decode(ids)

    def test_training_parameter_validation(self):
        for kwargs in (dict(text=""), dict(text="abc", vocab_size=2),
                       dict(text="ab", vocab_size=True), dict(text="ab", min_frequency=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BPETokenizer(**kwargs)
        with self.assertRaises(TypeError):
            BPETokenizer(None)

    def test_save_load_and_generic_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            for tokenizer in (self.tokenizer, CharTokenizer(self.text)):
                tokenizer.save(path)
                restored = load_tokenizer(path)
                self.assertIs(type(restored), type(tokenizer))
                self.assertEqual(restored.to_dict(), tokenizer.to_dict())
                self.assertEqual(restored.encode(self.text), tokenizer.encode(self.text))
            self.tokenizer.save(path)
            self.assertEqual(BPETokenizer.load(path).to_dict(), self.tokenizer.to_dict())

    def test_load_preserves_unsorted_base_ids(self):
        data = {"type": "bpe", "version": 1, "pre_tokenizer": "unicode_groups_v1",
                "chars": ["z", "a"], "merges": [[0, 1]]}
        tokenizer = BPETokenizer.from_dict(data)
        self.assertEqual(tokenizer.encode("za"), [2])
        self.assertEqual(tokenizer.encode("az"), [1, 0])

    def test_corrupt_merge_rules_rejected(self):
        base = {"type": "bpe", "version": 1, "pre_tokenizer": "unicode_groups_v1",
                "chars": ["a", "b"], "merges": []}
        for changes in ({"merges": [[0, 2]]}, {"merges": [[0, 1], [0, 1]]},
                        {"merges": [[True, 0]]}, {"chars": ["a", "a"]},
                        {"version": 99}, {"pre_tokenizer": "unknown"}):
            data = copy.deepcopy(base)
            data.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                BPETokenizer.from_dict(data)
        with self.assertRaises(ValueError):
            tokenizer_from_dict({"type": "unknown"})

    def test_random_round_trips_and_frozen_vocab(self):
        rng = random.Random(42)
        alphabet = "abc中\t\r\n !"
        training = alphabet + "".join(rng.choices(alphabet, k=300))
        tokenizer = BPETokenizer(training, vocab_size=40)
        before = tokenizer.to_dict()
        for _ in range(100):
            text = "".join(rng.choices(alphabet, k=rng.randrange(100)))
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.to_dict(), before)


@unittest.skipUnless(importlib.util.find_spec("torch"), "未安装 PyTorch，跳过模型服务联调")
class TestBPEModelService(unittest.TestCase):
    def test_checkpoint_and_generation(self):
        import torch
        from tests.test_generation import FakeModel
        from ui.model_service import ModelService
        tokenizer = BPETokenizer("hello hello world", vocab_size=30)
        config = {"vocab_size": tokenizer.vocab_size, "context_length": 16}
        model = FakeModel(config["vocab_size"], config["context_length"])
        service = ModelService(model_factory=lambda c: FakeModel(c["vocab_size"], c["context_length"]), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain_best.pt"
            checkpoint = {"model_state_dict": model.state_dict(), "config": config,
                          "training_step": 0, "validation_loss": 1.0,
                          "tokenizer": tokenizer.to_dict()}
            torch.save(checkpoint, path)
            service.load_model(path)
            self.assertIsInstance(service.tokenizer, BPETokenizer)
            ordinary = service.generate("hello", max_new_tokens=3, strategy="greedy")
            cached = service.generate("hello", max_new_tokens=3, strategy="greedy", use_kv_cache=True)
            self.assertEqual(ordinary["text"], cached["text"])
            self.assertEqual(cached["generated_tokens"], 3)
            self.assertTrue(cached["text"].startswith("hello"))
            # B 也可以单独交付 tokenizer.json；checkpoint 不必内嵌词表。
            tokenizer_path = Path(directory) / "tokenizer.json"
            tokenizer.save(tokenizer_path)
            del checkpoint["tokenizer"]
            torch.save(checkpoint, path)
            service.load_model(path, tokenizer_path)
            self.assertEqual(service.tokenizer.to_dict(), tokenizer.to_dict())

    def test_same_alphabet_different_merges_rejected(self):
        import torch
        from tests.test_generation import FakeModel
        from ui.model_service import ModelService
        first = {"type": "bpe", "version": 1, "pre_tokenizer": "unicode_groups_v1",
                 "chars": ["a", "b", "c"], "merges": [[0, 1]]}
        second = dict(first, merges=[[1, 2]])
        with tempfile.TemporaryDirectory() as directory:
            path, vocab = Path(directory) / "model.pt", Path(directory) / "tokenizer.json"
            BPETokenizer.from_dict(second).save(vocab)
            torch.save({"config": {"vocab_size": 4, "context_length": 8},
                        "model_state_dict": FakeModel(4, 8).state_dict(),
                        "tokenizer": first, "training_step": 0, "validation_loss": 1.0}, path)
            service = ModelService(model_factory=lambda c: FakeModel(4, 8), device="cpu")
            with self.assertRaisesRegex(ValueError, "合并规则"):
                service.load_model(path, vocab)


if __name__ == "__main__":
    unittest.main()
