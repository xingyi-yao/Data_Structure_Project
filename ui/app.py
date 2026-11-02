"""三页 Gradio UI。根目录运行：python -m ui.app

先运行预训练/微调命令生成 checkpoint，再使用对应页面。
页面可先启动；缺少文件时点击按钮会显示错误提示，不伪造模型输出。
Gradio API 参考：https://www.gradio.app/docs/gradio/blocks
"""
from .pages.generation import build_generation_page
from .pages.finetune_demo import build_finetune_page
from .pages.benchmark import build_benchmark_page
from functools import wraps
import gradio as gr
from .model_service import (ModelService, MODEL_SPECS, PROJECT_ROOT,
                            compare_finetuning, save_generation_samples)
from src.optimization.benchmark import save_results, plot_results


def user_error(fn):
    """callback 出错时在页面显示可读提示，后台服务继续运行。"""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


def create_app():
    # 本地课程展示服务；所有按钮共用同一并发组，避免显存竞争干扰 benchmark。
    services = {}

    def get_service(name):
        if name not in MODEL_SPECS:
            raise ValueError("请选择 pretrain 或 finetune")
        if name not in services:
            service = ModelService()
            spec = MODEL_SPECS[name]
            service.load_model(spec["checkpoint"], spec["tokenizer"])
            services[name] = service
        return services[name]

    def settings(count, temperature, top_k, strategy, cache, seed):
        # UI 约定 top_k=0 表示关闭，传入底层时转换成文档的 None。
        return dict(max_new_tokens=int(count), temperature=float(temperature),
                    top_k=None if int(top_k) == 0 else int(top_k),
                    strategy=strategy, use_kv_cache=bool(cache), seed=int(seed))

    @user_error
    def do_generate(name, prompt, count, temperature, top_k, strategy, cache, seed):
        result = get_service(name).generate(prompt, **settings(count, temperature, top_k, strategy, cache, seed))
        save_generation_samples([result])
        stats = {k: result[k] for k in ("generated_tokens", "seconds", "tokens_per_second", "checkpoint")}
        return result["text"], stats

    @user_error
    def do_compare(prompt, count, temperature, top_k, strategy, cache, seed):
        result = compare_finetuning(get_service("pretrain"), get_service("finetune"), prompt,
                                   **settings(count, temperature, top_k, strategy, cache, seed))
        save_generation_samples(list(result.values()))
        stats = {name: {k: row[k] for k in ("seconds", "tokens_per_second", "checkpoint")}
                 for name, row in result.items()}
        return result["pretrain"]["text"], result["finetune"]["text"], stats

    @user_error
    def do_benchmark(name, prompt, length_text, repeats):
        # 输入多个长度就运行完整曲线；默认是文档的 50/100/200/500。
        lengths = [int(value.strip()) for value in length_text.replace("，", ",").split(",")]
        if not lengths or any(n <= 0 for n in lengths):
            raise ValueError("长度请填写正整数，用逗号分隔")
        service = get_service(name)
        comparisons = [service.benchmark(prompt, n, repeats=int(repeats)) for n in lengths]
        rows = [r[key] for r in comparisons for key in ("no_cache", "with_cache")]
        directory = PROJECT_ROOT / "outputs/benchmark"
        save_results(rows, directory)
        save_results(comparisons, directory, "kv_cache_comparison")
        figure = plot_results(rows, directory / "benchmark_curve.png")
        table = [[r["max_new_tokens"], r["no_cache"]["median_seconds"],
                  r["with_cache"]["median_seconds"], r["no_cache"]["tokens_per_second"],
                  r["with_cache"]["tokens_per_second"], r["speedup"]] for r in comparisons]
        return table, figure, comparisons

    with gr.Blocks(title="Small Language Model") as demo:
        gr.Markdown("# Small Language Model\n文本生成 · 微调对比 · KV Cache 性能测试")
        build_generation_page(do_generate)
        build_finetune_page(do_compare)
        build_benchmark_page(do_benchmark)
    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1", share=False)
