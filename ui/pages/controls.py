import gradio as gr
from ui.config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED

def generation_controls():
    count = gr.Slider(1, 500, value=DEFAULT_MAX_NEW_TOKENS, step=1, label="Max New Tokens")
    temperature = gr.Slider(0.05, 2.0, value=1.0, step=0.05, label="Temperature")
    top_k = gr.Number(value=0, minimum=0, precision=0, label="Top-k（0 表示关闭）")
    strategy = gr.Radio(["sample", "greedy"], value="sample", label="Sampling")
    cache = gr.Checkbox(value=False, label="Use KV Cache")
    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Random Seed")
    return [count, temperature, top_k, strategy, cache, seed]
