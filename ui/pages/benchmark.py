import gradio as gr
from .controls import generation_controls
from ui.config import DEFAULT_BENCHMARK_LENGTHS


def build_benchmark_page(do_benchmark):
    with gr.Tab("Performance Benchmark"):
        gr.Markdown("固定使用 Greedy，先检查缓存前后生成一致，再进行预热与重复计时。")
        name = gr.Dropdown(["pretrain", "finetune"], value="pretrain", label="Model")
        prompt = gr.Textbox(label="Prompt")
        lengths = gr.Textbox(value=DEFAULT_BENCHMARK_LENGTHS, label="生成长度")
        repeats = gr.Slider(1, 10, value=3, step=1, label="重复次数")
        button = gr.Button("Run Benchmark")
        table = gr.Dataframe(headers=["Tokens", "No Cache 秒", "Cache 秒",
                                      "No Cache tok/s", "Cache tok/s", "Speedup"],
                             datatype=["number"] * 6, interactive=False)
        figure = gr.Image(type="filepath", label="实测曲线")
        raw = gr.JSON(label="原始记录与内存统计")
        button.click(do_benchmark, [name, prompt, lengths, repeats], [table, figure, raw],
                     concurrency_id="model_operations", concurrency_limit=1)
