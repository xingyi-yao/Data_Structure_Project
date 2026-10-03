"""Baseline Benchmark 和公共计时工具。

从项目根目录运行：
python -m src.optimization.benchmark --prompt "hello" --lengths 50 100 200 500
TODO(A/B)：先完成 ui/model_service.py 中的模型构造入口和 checkpoint/词表路径。
CPU RSS 是进程内存快照，不声称是每个算子的内存峰值。
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from src.inference.generate import generate, get_context_length


def synchronize(device):
    """CUDA 异步执行，计时前后都要等设备完成。"""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def seeded(seed, device):
    """固定 CPU/当前 CUDA 随机种子，并在退出时恢复调用方的随机状态。"""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


def precision_context(device, precision="fp32"):
    """模型权重保持 FP32；CUDA 下通过 autocast 测 FP16/BF16 混合精度。"""
    device = torch.device(device)
    if precision == "fp32":
        return nullcontext()
    if precision not in ("fp16", "bf16") or device.type != "cuda":
        raise ValueError("FP16/BF16 实验仅支持 CUDA；CPU 请用 fp32")
    if precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("当前 GPU 不支持 BF16")
    return torch.autocast(device_type="cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def cpu_rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except (psutil.Error, OSError):
        # 某些沙箱/系统不开放进程内存信息；缺失就记为 None，不编造 0 MB。
        return None


def measure_callable(fn, device, *, warmup=1, repeats=3) -> dict:
    """warm-up 不计入结果；记录每次原始数据，汇总采用中位数。"""
    device = torch.device(device)
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmup >= 0，repeats >= 1，均为整数")
    for _ in range(warmup):
        value = fn()
        synchronize(device)
        del value
    rows = []
    for _ in range(repeats):
        synchronize(device)
        rss_before = cpu_rss_mb()
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = None
        start = time.perf_counter()
        value = fn()
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
        rows.append({"seconds": elapsed, "cpu_rss_before_mb": rss_before,
                     "cpu_rss_after_mb": cpu_rss_mb(), "cuda_allocated_before_mb": baseline,
                     "cuda_peak_allocated_mb": peak,
                     "cuda_peak_extra_mb": peak - baseline if peak is not None else None})
        del value
    return {"median_seconds": statistics.median(r["seconds"] for r in rows),
            "min_seconds": min(r["seconds"] for r in rows),
            "warmup": warmup, "repeats": repeats, "runs": rows}


def benchmark_generation(model, prompt, max_new_tokens, use_kv_cache=False, *,
                         tokenizer=None, device=None, temperature=1.0, top_k=None,
                         strategy="greedy", context_length=None, warmup=1, repeats=3,
                         seed=42, precision="fp32", model_label="unspecified") -> dict:
    """prompt 可传 [B,T] token Tensor；传字符串时同时传 tokenizer。"""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("Benchmark 的 max_new_tokens 必须为正整数")
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("字符串 prompt 需要 tokenizer")
        if device is None:
            device = next(model.parameters()).device
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    elif isinstance(prompt, torch.Tensor):
        ids = prompt if device is None else prompt.to(device)
    else:
        raise TypeError("prompt 必须是字符串或 [B,T] Tensor")
    if ids.ndim != 2 or min(ids.shape) < 1:
        raise ValueError("Benchmark prompt 必须为非空 [B,T]")
    device = ids.device
    limit = get_context_length(model, context_length)

    def run():
        with seeded(seed, device), precision_context(device, precision):
            return generate(model, ids, max_new_tokens, temperature, top_k, use_kv_cache,
                            context_length=limit, strategy=strategy)

    measured = measure_callable(run, device, warmup=warmup, repeats=repeats)
    count = ids.size(0) * max_new_tokens  # batch 内实际新增总数，不计 prompt。
    measured.update({"model": model_label, "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_num_threads": torch.get_num_threads(), "batch_size": ids.size(0),
        "prompt_tokens_per_sequence": ids.size(1), "max_new_tokens": max_new_tokens,
        "generated_tokens_total": count, "tokens_per_second": count / measured["median_seconds"],
        "use_kv_cache": use_kv_cache, "context_length": limit, "strategy": strategy,
        "temperature": temperature, "top_k": top_k, "seed": seed, "precision": precision,
        "window_rebuild_required": ids.size(1) + max_new_tokens - 1 > limit,
        "memory_note": "CPU RSS 为前后快照；CUDA peak 含已驻留权重；extra 为本轮峰值增量"})
    return measured


def save_results(results, directory="outputs/benchmark", stem="benchmark_results"):
    """保存 JSON 原始记录、可读 TXT 和 CSV。目录运行时自动创建。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, ensure_ascii=False, indent=2)
    json_path, txt_path = directory / f"{stem}.json", directory / f"{stem}.txt"
    json_path.write_text(text, encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")
    flat_rows = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in results]
    csv_path = directory / f"{stem}.csv"
    fields = list(dict.fromkeys(k for row in flat_rows for k in row))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    return [str(json_path), str(txt_path), str(csv_path)]


def plot_results(results, path="outputs/benchmark/benchmark_curve.png"):
    """绘制实测耗时/吞吐曲线；不会预填性能数字。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = {}
    for row in results:
        label = f"{'KV cache' if row['use_kv_cache'] else 'No cache'} / {row['precision']}"
        groups.setdefault(label, []).append(row)
    for label, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["max_new_tokens"])
        x = [r["max_new_tokens"] for r in rows]
        axes[0].plot(x, [r["median_seconds"] for r in rows], "o-", label=label)
        axes[1].plot(x, [r["tokens_per_second"] for r in rows], "o-", label=label)
    for ax, y in zip(axes, ("Time (s)", "Tokens / second")):
        ax.set(xlabel="New tokens per sequence", ylabel=y)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from ui.model_service import ModelService, MODEL_SPECS
    service = ModelService(device=args.device)
    spec = MODEL_SPECS[args.model]
    service.load_model(spec["checkpoint"], spec["tokenizer"])
    rows = [benchmark_generation(service.model, args.prompt, n,
        tokenizer=service.tokenizer, device=service.device, precision=args.precision,
        repeats=args.repeats, context_length=service.context_length,
        model_label=str(service.checkpoint_path)) for n in args.lengths]
    save_results(rows)
    plot_results(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
