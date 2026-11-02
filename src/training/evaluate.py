import math
import torch
from src.inference.generate import evaluating


def evaluate(model, dataset, batch_size=16, eval_batches=20, device=None, seed=1234):
    """固定评估窗口，不消耗训练随机数；返回平均 token 损失与困惑度。"""
    if eval_batches < 1:
        raise ValueError("eval_batches 必须大于零")
    device = device or next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    losses = []
    with evaluating(model):
        for _ in range(eval_batches):
            x, y = dataset.get_batch(batch_size, device, generator)
            _, loss = model(x, y)
            losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 700 else float("inf")}


def estimate_loss(model, datasets, **kwargs):
    return {name: evaluate(model, dataset, **kwargs)["loss"] for name, dataset in datasets.items()}
