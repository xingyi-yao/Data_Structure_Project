"""模型 shape/loss/因果性 + Checkpoint 加载测试。

服务层使用临时假 checkpoint 测流程；真实 GPT 的测试需先填写 TODO(A)。
未接 A 的模型时用 skip 明确报告，不把跳过算作真实模型通过。
"""
import tempfile
import unittest
from pathlib import Path
import torch

from tests.test_generation import FakeModel
from ui.model_service import ModelService, build_model_from_config, compare_finetuning
from src.tokenizer.char_tokenizer import CharTokenizer
from src.inference.generate import generate


# TODO(A)：将这组小尺寸测试配置的键名与 A 的构造函数统一。
SMALL_CONFIG = dict(vocab_size=7, context_length=16, embedding_dim=16,
                    num_heads=2, num_layers=2, dropout=0.0)


class TestModelService(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "pretrain_best.pt"
        self.tokenizer = CharTokenizer("abcdefg")
        self.model = FakeModel(7, 16)
        torch.save({"model_state_dict": self.model.state_dict(), "config": SMALL_CONFIG,
                    "training_step": 0, "validation_loss": 1.0,
                    "tokenizer": self.tokenizer.to_dict()}, self.path)
        self.service = ModelService(
            model_factory=lambda config: FakeModel(config["vocab_size"], config["context_length"]),
            device="cpu")

    def test_load_generate_and_compare(self):
        metadata = self.service.load_model(self.path)
        self.assertEqual(metadata["vocab_size"], 7)
        self.assertFalse(self.service.model.training)
        result = self.service.generate("ab", 3, strategy="greedy", use_kv_cache=True)
        self.assertEqual(result["generated_tokens"], 3)
        self.assertTrue(result["text"].startswith("ab"))
        result = compare_finetuning(self.service, self.service, "ab",
                                   max_new_tokens=3, strategy="sample", seed=42)
        self.assertEqual(result["pretrain"]["text"], result["finetune"]["text"])

    def test_vocab_mapping_mismatch(self):
        mismatched = CharTokenizer.from_dict({"type": "char", "version": 1,
                                              "chars": list("gfedcba")})
        path = Path(self.directory.name) / "tokenizer.json"
        mismatched.save(path)
        with self.assertRaisesRegex(ValueError, "ID 顺序"):
            self.service.load_model(self.path, path)

    def test_missing_checkpoint_keeps_loaded_model(self):
        self.service.load_model(self.path)
        old_model = self.service.model
        with self.assertRaises(FileNotFoundError):
            self.service.load_model(Path(self.directory.name) / "absent.pt")
        self.assertIs(old_model, self.service.model)

    def test_missing_and_unknown_prompt(self):
        with self.assertRaises(RuntimeError):
            self.service.generate("a")
        self.service.load_model(self.path)
        with self.assertRaises(ValueError):
            self.service.generate("")
        with self.assertRaises(ValueError):
            self.service.generate("z")


class TestRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = build_model_from_config(SMALL_CONFIG).cpu().eval()
        except NotImplementedError as exc:
            raise unittest.SkipTest(str(exc))
        # 只捕获尚未接入的 NotImplementedError；A 的实现错误应正常导致失败。

    def test_shape_and_scalar_loss(self):
        idx = torch.randint(0, 7, (2, 8))
        targets = torch.randint(0, 7, (2, 8))
        with torch.no_grad():
            logits, loss = self.model(idx, targets=targets)
        self.assertEqual(logits.shape, (2, 8, 7))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_no_future_information(self):
        idx = torch.tensor([[0, 1, 2, 3, 4, 5]])
        changed = idx.clone()
        changed[:, 3:] = torch.tensor([6, 0, 1])
        with torch.no_grad():
            first, _ = self.model(idx)
            second, _ = self.model(changed)
        torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-5, rtol=1e-4)

    def test_real_cache_logits_and_generation(self):
        idx = torch.tensor([[0, 1, 2, 3]])
        with torch.no_grad():
            full, _ = self.model(idx)
            _, cache = self.model(idx[:, :3], past_kv=None, use_cache=True)
            last, _ = self.model(idx[:, 3:], past_kv=cache, use_cache=True)
        torch.testing.assert_close(full[:, -1:], last, atol=1e-5, rtol=1e-4)
        for count in (4, 20):  # 短生成及超窗口生成均检查。
            first = generate(self.model, idx, count, strategy="greedy", context_length=16)
            second = generate(self.model, idx, count, strategy="greedy", use_kv_cache=True,
                              context_length=16)
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
