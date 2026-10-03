"""覆盖缓存传递、长度增长、逐层独立、滑动窗口与普通生成等价。"""
import unittest
import torch
from tests.test_generation import FakeModel
from src.inference.generate import generate
from src.inference.kv_cache import validate_cache, append_layer_kv, cache_nbytes
from src.optimization.kv_cache_benchmark import compare_kv_cache


class TestKVCache(unittest.TestCase):
    def test_prefill_then_single_token(self):
        model = FakeModel(7, 16)
        prompt = torch.tensor([[0, 1], [2, 3]])
        baseline = generate(model, prompt, 5, strategy="greedy")
        model.seen_lengths.clear()
        model.seen_past.clear()
        cached = generate(model, prompt, 5, use_kv_cache=True, strategy="greedy")
        self.assertTrue(torch.equal(baseline, cached))
        self.assertEqual(model.seen_lengths, [2, 1, 1, 1, 1])
        self.assertEqual(model.seen_past, [0, 2, 3, 4, 5])

    def test_overflow_rebuild_and_new_request(self):
        prompt = torch.tensor([[0, 1, 2, 3, 4]])
        model = FakeModel(7, 3)
        baseline = generate(model, prompt, 8, strategy="greedy")
        cached = generate(model, prompt, 8, use_kv_cache=True, strategy="greedy")
        self.assertTrue(torch.equal(baseline, cached))
        self.assertTrue(all(length <= 3 for length in model.seen_lengths))
        again = generate(model, prompt, 8, use_kv_cache=True, strategy="greedy")
        self.assertTrue(torch.equal(cached, again))

    def test_layer_cache_and_append(self):
        key, value = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
        new_key, new_value = torch.randn(2, 3, 1, 5), torch.randn(2, 3, 1, 5)
        merged = append_layer_kv((key, value), new_key, new_value)
        self.assertEqual(merged[0].shape, (2, 3, 5, 5))
        torch.testing.assert_close(merged[0][:, :, :4], key)
        self.assertEqual(validate_cache([merged], batch_size=2, expected_length=5), 5)
        self.assertEqual(cache_nbytes([merged]), 2 * 2 * 3 * 5 * 5 * key.element_size())
        model = FakeModel()
        _, cache = model(torch.tensor([[1, 2]]), use_cache=True)
        self.assertNotEqual(cache[0][0].data_ptr(), cache[1][0].data_ptr())

    def test_invalid_cache_is_detected(self):
        with self.assertRaises(ValueError):
            validate_cache([])
        with self.assertRaises(ValueError):
            validate_cache([(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))])
        with self.assertRaises(ValueError):
            validate_cache([(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3))],
                           expected_length=3)

    def test_bad_model_cache_length(self):
        class Broken(FakeModel):
            def forward(self, *args, **kwargs):
                output, cache = super().forward(*args, **kwargs)
                if kwargs.get("use_cache"):
                    cache = [(k[:, :, :1], v[:, :, :1]) for k, v in cache]
                return output, cache
        model = Broken()
        with self.assertRaises(ValueError):
            generate(model, torch.tensor([[0, 1]]), 2, use_kv_cache=True)
        self.assertTrue(model.training)  # 异常路径也应恢复训练状态。

    def test_benchmark_plumbing(self):
        # 只测试计时代码的数据流；FakeModel 的速度不作为项目实验结论。
        result = compare_kv_cache(FakeModel(), torch.tensor([[0, 1]]), 2, repeats=1, warmup=0)
        self.assertTrue(result["greedy_equal"])
        self.assertGreater(result["speedup"], 0)
        self.assertEqual(result["no_cache"]["generated_tokens_total"], 2)


if __name__ == "__main__":
    unittest.main()
