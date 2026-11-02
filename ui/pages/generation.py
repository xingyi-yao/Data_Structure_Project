import gradio as gr
from .controls import generation_controls
from ui.config import DEFAULT_BENCHMARK_LENGTHS


def build_generation_page(do_generate):
    with gr.Tab("Text Generation"):
        name = gr.Dropdown(["pretrain", "finetune"], value="pretrain", label="Model")
        prompt = gr.Textbox(label="Prompt", lines=3)
        controls = generation_controls()
        button = gr.Button("Generate")
        output = gr.Textbox(label="Generated Text", lines=10)
        stats = gr.JSON(label="生成时间与速度")
        button.click(do_generate, [name, prompt, *controls], [output, stats],
                     concurrency_id="model_operations", concurrency_limit=1)
