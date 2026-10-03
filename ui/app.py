"""三页 Gradio UI。根目录运行：python -m ui.app

TODO(A/B)：先填写 model_service.py 的模型构造入口与 checkpoint/词表路径。
页面可先启动；缺少文件时点击按钮会显示错误提示，不伪造模型输出。
Gradio API 参考：https://www.gradio.app/docs/gradio/blocks
"""
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

    def generation_controls():
        count = gr.Slider(1, 500, value=100, step=1, label="Max New Tokens")
        temperature = gr.Slider(0.05, 2.0, value=1.0, step=0.05, label="Temperature")
        top_k = gr.Number(value=0, minimum=0, precision=0, label="Top-k（0 表示关闭）")
        strategy = gr.Radio(["sample", "greedy"], value="sample", label="Sampling")
        cache = gr.Checkbox(value=False, label="Use KV Cache")
        seed = gr.Number(value=42, precision=0, label="Random Seed")
        return [count, temperature, top_k, strategy, cache, seed]

    with gr.Blocks(title="Small Language Model") as demo:
        gr.Markdown("# Small Language Model\n文本生成 · 微调对比 · KV Cache 性能测试")
        with gr.Tab("Text Generation"):
            name = gr.Dropdown(["pretrain", "finetune"], value="pretrain", label="Model")
            prompt = gr.Textbox(label="Prompt", lines=3)
            controls = generation_controls()
            button = gr.Button("Generate")
            output = gr.Textbox(label="Generated Text", lines=10)
            stats = gr.JSON(label="生成时间与速度")
            button.click(do_generate, [name, prompt, *controls], [output, stats],
                         concurrency_id="model_operations", concurrency_limit=1)
        with gr.Tab("Fine-Tuning Comparison"):
            prompt = gr.Textbox(label="同一个 Prompt", lines=3)
            controls = generation_controls()
            button = gr.Button("Compare")
            with gr.Row():
                before = gr.Textbox(label="Pre-trained Output", lines=10)
                after = gr.Textbox(label="Fine-tuned Output", lines=10)
            stats = gr.JSON(label="运行信息")
            button.click(do_compare, [prompt, *controls], [before, after, stats],
                         concurrency_id="model_operations", concurrency_limit=1)
        with gr.Tab("Performance Benchmark"):
            gr.Markdown("固定使用 Greedy，先检查缓存前后生成一致，再进行预热与重复计时。")
            name = gr.Dropdown(["pretrain", "finetune"], value="pretrain", label="Model")
            prompt = gr.Textbox(label="Prompt")
            lengths = gr.Textbox(value="50,100,200,500", label="生成长度")
            repeats = gr.Slider(1, 10, value=3, step=1, label="重复次数")
            button = gr.Button("Run Benchmark")
            table = gr.Dataframe(headers=["Tokens", "No Cache 秒", "Cache 秒",
                                          "No Cache tok/s", "Cache tok/s", "Speedup"],
                                 datatype=["number"] * 6, interactive=False)
            figure = gr.Image(type="filepath", label="实测曲线")
            raw = gr.JSON(label="原始记录与内存统计")
            button.click(do_benchmark, [name, prompt, lengths, repeats], [table, figure, raw],
                         concurrency_id="model_operations", concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1", share=False)
