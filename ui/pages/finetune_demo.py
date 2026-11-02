import gradio as gr
from .controls import generation_controls
from ui.config import DEFAULT_BENCHMARK_LENGTHS


def build_finetune_page(do_compare):
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
