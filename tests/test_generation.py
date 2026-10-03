"""用确定性 FakeModel 测试 C 的数据流；它不是 A 的 GPT，也不代表真实模型质量。"""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from src.inference.generate import generate
from src.inference.sampling import sample_next_token
from src.tokenizer.char_tokenizer import CharTokenizer


class FakeModel(torch.nn.Module):
    """测试替身：下一 token=(本轮窗口内 token ID 累加和+位置) mod V。

    假缓存里保存 ID，仅用于检查 C 是否正确传递/重置历史；
    不可用于实际 KV Cache 性能结论。
    """
    def __init__(self, vocab_size=7, context_length=8):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, context_length=context_length)
        self.register_buffer("anchor", torch.zeros(1))
        self.seen_lengths, self.seen_past, self.saw_training, self.saw_grad = [], [], [], []

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        if idx.device != self.anchor.device:
            raise ValueError("输入与模型 device 不一致")
        self.seen_lengths.append(idx.size(1))
        self.seen_past.append(0 if past_kv is None else past_kv[0][0].size(-2))
        self.saw_training.append(self.training)
        self.saw_grad.append(torch.is_grad_enabled())
        old = None if past_kv is None else past_kv[0][0][:, 0, :, 0].long()
        full = idx if old is None else torch.cat((old, idx), dim=1)
        if full.size(1) > self.config.context_length:
            raise ValueError("超过模型 context_length")
        position = torch.arange(full.size(1), device=idx.device)
        expected = (full.cumsum(dim=1) + position) % self.config.vocab_size
        logits = F.one_hot(expected[:, -idx.size(1):], self.config.vocab_size).float() * 10
        if use_cache:
            key = full.float()[:, None, :, None]
            # 独立的两层缓存，防止 tests 把所有层指向同一 Tensor。
            return logits, [(key.clone(), key.clone()), (key.clone() + 1, key.clone() + 2)]
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class TestGeneration(unittest.TestCase):
    def test_pipeline_batch_and_context(self):
        tokenizer = CharTokenizer("abcde")
        ids = torch.tensor([tokenizer.encode("ab"), tokenizer.encode("cd")])
        model = FakeModel(5, 3)
        result = generate(model, ids, 4, strategy="greedy")
        self.assertEqual(result.shape, (2, 6))
        self.assertEqual(result[0].tolist(), [0, 1, 2, 0, 0, 4])
        self.assertEqual(tokenizer.decode(result[0].tolist()), "abcaae")
        self.assertEqual(model.seen_lengths, [2, 3, 3, 3])
        self.assertTrue(torch.equal(result[:, :2], ids))
        self.assertTrue(((result >= 0) & (result < 5)).all().item())
        self.assertTrue(model.training)  # 调用前处于训练模式；调用后恢复。
        self.assertFalse(any(model.saw_training))
        self.assertFalse(any(model.saw_grad))

    def test_zero_tokens(self):
        ids = torch.tensor([[0, 1]])
        self.assertTrue(torch.equal(generate(FakeModel(), ids, 0), ids))

    def test_sampling_modes(self):
        logits = torch.tensor([[0., 2., 1.], [5., 1., 0.]])
        self.assertEqual(sample_next_token(logits, strategy="greedy").tolist(), [[1], [0]])
        self.assertEqual(sample_next_token(logits, temperature=0.5, top_k=1).tolist(), [[1], [0]])
        torch.manual_seed(42)
        many = logits[:1].repeat(64, 1)
        ids = sample_next_token(many, temperature=1.5, top_k=2)
        self.assertTrue(((ids == 1) | (ids == 2)).all().item())
        self.assertEqual(sample_next_token(logits, top_k=None).shape, (2, 1))
        # 并列分数只保留一个候选，不能把两个并列第一都留下。
        tied = sample_next_token(torch.zeros(50, 3), top_k=1)
        self.assertEqual(tied.unique().numel(), 1)

    def test_invalid_inputs(self):
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                sample_next_token(torch.zeros(1, 3), temperature)
        with self.assertRaises(ValueError):
            sample_next_token(torch.zeros(1, 3), top_k=4)
        with self.assertRaises(ValueError):
            generate(FakeModel(), torch.empty(1, 0, dtype=torch.long))
        with self.assertRaises(TypeError):
            generate(FakeModel(), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            sample_next_token(torch.full((1, 3), float("-inf")))

    @unittest.skipUnless(torch.cuda.is_available(), "无 CUDA，跳过 GPU 测试")
    def test_cuda(self):
        result = generate(FakeModel().cuda(), torch.tensor([[0, 1]], device="cuda"), 5, strategy="greedy")
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
