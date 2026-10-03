"""同一模型、输入和参数下比较 No Cache / KV Cache；先验正确性，再计时。"""
import argparse
import json
import torch
from src.inference.generate import generate, get_context_length
from .benchmark import benchmark_generation, save_results, plot_results


def compare_kv_cache(model, input_ids, max_new_tokens, *, context_length=None,
                     repeats=3, warmup=1, seed=42, precision="fp32",
                     model_label="unspecified"):
    """使用 Greedy 排除采样噪声；输出相同才汇报性能对比。"""
    limit = get_context_length(model, context_length)
    from .benchmark import precision_context
    with precision_context(input_ids.device, precision):
        ordinary = generate(model, input_ids, max_new_tokens,
                            strategy="greedy", context_length=limit)
        cached = generate(model, input_ids, max_new_tokens, use_kv_cache=True,
                          strategy="greedy", context_length=limit)
    if not torch.equal(ordinary, cached):
        raise ValueError("KV Cache 与普通 Greedy 生成不一致，请先检查 A 的位置编码、mask 与缓存")
    del ordinary, cached
    common = dict(context_length=limit, repeats=repeats, warmup=warmup,
                  seed=seed, precision=precision, model_label=model_label)
    # 两种模式每次都从同一 input_ids 开始；generate 内部缓存不跨请求共享。
    no_cache = benchmark_generation(model, input_ids, max_new_tokens, False, **common)
    with_cache = benchmark_generation(model, input_ids, max_new_tokens, True, **common)
    return {"max_new_tokens": max_new_tokens, "greedy_equal": True,
            "speedup": no_cache["median_seconds"] / with_cache["median_seconds"],
            "no_cache": no_cache, "with_cache": with_cache,
            "note": "窗口满后重建缓存，长生成不保证加速；数值为当前机器实测"}


def run_kv_cache_sweep(model, input_ids, lengths=(50, 100, 200, 500), *,
                       output_dir="outputs/benchmark", **kwargs):
    comparisons = [compare_kv_cache(model, input_ids, n, **kwargs) for n in lengths]
    rows = [result[mode] for result in comparisons for mode in ("no_cache", "with_cache")]
    save_results(comparisons, output_dir, "kv_cache_comparison")
    save_results(rows, output_dir, "benchmark_results")
    plot_results(rows, f"{output_dir}/benchmark_curve.png")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from ui.model_service import ModelService, MODEL_SPECS
    service = ModelService(device=args.device)
    spec = MODEL_SPECS[args.model]
    service.load_model(spec["checkpoint"], spec["tokenizer"])
    ids = service.encode_prompt(args.prompt)
    results = run_kv_cache_sweep(service.model, ids, args.lengths,
        context_length=service.context_length, repeats=args.repeats,
        precision=args.precision, model_label=str(service.checkpoint_path))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
