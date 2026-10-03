"""Attention shape 与因果性；A 未接入时真实模块测试会显示 skipped。"""
import unittest
import torch
import torch.nn.functional as F
from src.optimization.attention_benchmark import manual_attention, build_attention_variants
from src.inference.generate import evaluating


class TestAttentionReference(unittest.TestCase):
    def test_manual_matches_sdpa(self):
        torch.manual_seed(42)
        q, k, v = [torch.randn(2, 3, 8, 4) for _ in range(3)]
        manual = manual_attention(q, k, v)
        sdpa = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        self.assertEqual(manual.shape, q.shape)
        torch.testing.assert_close(manual, sdpa, atol=1e-5, rtol=1e-4)

    def test_reference_causality(self):
        torch.manual_seed(42)
        q, k, v = [torch.randn(2, 2, 6, 4) for _ in range(3)]
        before = manual_attention(q, k, v)
        k[:, :, 3:] += 10
        v[:, :, 3:] -= 10
        after = manual_attention(q, k, v)
        torch.testing.assert_close(before[:, :, :3], after[:, :, :3])


class TestRealAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.variants = build_attention_variants()
        except NotImplementedError as exc:
            raise unittest.SkipTest(str(exc))

    def test_shape_causality_and_equivalence(self):
        # TODO(A)：根据 build_attention_variants 的配置调整 C=128；
        # T=16 应不超过模型 context_length，B=2。
        x = torch.randn(2, 16, 128)
        altered = x.clone()
        altered[:, 8:] = torch.randn_like(altered[:, 8:])
        reference = None
        for name, module in self.variants.items():
            with self.subTest(variant=name), evaluating(module.cpu()):
                y, z = module(x), module(altered)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all().item())
                torch.testing.assert_close(y[:, :8], z[:, :8], atol=1e-5, rtol=1e-4)
                if reference is None:
                    reference = y
                else:
                    torch.testing.assert_close(y, reference, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
