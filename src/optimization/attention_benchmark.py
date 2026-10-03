"""Attention 性能对照：手写注意力核 vs SDPA，以及 A 提供的模型变体。

独立运行：python -m src.optimization.attention_benchmark --lengths 64 128 256
这个独立实验只比较同一 Q/K/V 的注意力计算，不代表整个 GPT 的加速比。
TODO(A)：预生成 mask、合并 QKV 的 Attention 模块接到 build_attention_variants。
官方 SDPA API：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
"""
import argparse
import json
import math
import torch
import torch.nn.functional as F
from src.inference.generate import evaluating
from .benchmark import measure_callable, precision_context, save_results, seeded


def manual_attention(q, k, v):
    """手写参考核，仅用于 benchmark。q/k/v=[B,H,T,D]。"""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.ones(q.size(-2), k.size(-2), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def benchmark_attention_kernels(lengths=(64, 128, 256), *, batch_size=2,
                                num_heads=4, head_dim=32, device="cpu",
                                precision="fp32", warmup=2, repeats=5):
    device = torch.device(device)
    if any(type(n) is not int or n < 1 for n in (*lengths, batch_size, num_heads, head_dim)):
        raise ValueError("所有 shape 维度必须为正整数")
    rows = []
    for length in lengths:
        with seeded(42, device):
            q, k, v = [torch.randn(batch_size, num_heads, length, head_dim, device=device) for _ in range(3)]
        with torch.inference_mode(), precision_context(device, precision):
            reference = manual_attention(q, k, v)
            optimized = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            tolerance = 1e-4 if precision == "fp32" else 3e-2
            torch.testing.assert_close(reference, optimized, atol=tolerance, rtol=tolerance)
            error = (reference - optimized).abs().max().item()
        del reference, optimized
        for name, kernel in (("manual", manual_attention),
                             ("sdpa", lambda q, k, v: F.scaled_dot_product_attention(
                                 q, k, v, dropout_p=0.0, is_causal=True))):
            def run():
                with torch.inference_mode(), precision_context(device, precision):
                    return kernel(q, k, v)
            result = measure_callable(run, device, warmup=warmup, repeats=repeats)
            result.update({"variant": name, "sequence_length": length, "batch_size": batch_size,
                           "num_heads": num_heads, "head_dim": head_dim, "device": str(device),
                           "precision": precision, "max_abs_error": error, "torch_version": str(torch.__version__)})
            rows.append(result)
    return rows


def build_attention_variants():
    """TODO(A)：此处返回你们实际模块，C 无法替 A 猜构造函数和参数名。

    示例（根据 A 的类名修改并取消注释）：
        from src.model.attention import ManualAttention, OptimizedAttention
        baseline = ManualAttention(config)
        optimized = OptimizedAttention(config)
        # 由 A 显式拷贝同一套权重；合并 QKV 时 state_dict 名称通常不同。
        # optimized.qkv.weight.copy_(torch.cat([baseline.query.weight, ...], dim=0))
        return {"manual": baseline, "precomputed_mask": optimized}

    每个模块调用 module(x) -> output Tensor，shape=[B,T,C]。
    若 A 的模块返回 (output, cache)，在此处用包装模块取 result[0]。
    """
    raise NotImplementedError("TODO(A)：接入具有相同权重的 Attention 变体")


def benchmark_attention_variants(variants, x, *, warmup=2, repeats=5,
                                 atol=1e-5, rtol=1e-4):
    """可测预生成 mask、合并 QKV 等；先确保输出与第一个基线模块等价。"""
    if not variants:
        raise ValueError("variants 不能为空")
    reference = None
    rows = []
    for name, module in variants.items():
        with evaluating(module):
            output = module(x)
            if not isinstance(output, torch.Tensor) or output.shape != x.shape:
                raise ValueError("Attention 模块应返回与 x 同 shape 的 Tensor")
            if not torch.isfinite(output).all():
                raise ValueError("Attention 输出含 NaN/Inf")
            if reference is None:
                reference = output.detach().clone()
            torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
            error = (output - reference).abs().max().item()
        del output
        def run():
            with evaluating(module):
                return module(x)
        row = measure_callable(run, x.device, warmup=warmup, repeats=repeats)
        row.update({"variant": name, "max_abs_error": error, "input_shape": list(x.shape),
                    "device": str(x.device)})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()
    rows = benchmark_attention_kernels(args.lengths, device=args.device, precision=args.precision)
    save_results(rows, stem="attention_benchmark")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
