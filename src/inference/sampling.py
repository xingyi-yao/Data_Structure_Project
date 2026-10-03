"""Sampling：logits [B,V] -> next token [B,1]。"""
import math
import torch


def validate_sampling(temperature=1.0, top_k=None, strategy="sample", vocab_size=None):
    if strategy not in ("greedy", "sample"):
        raise ValueError("strategy 必须为 'greedy' 或 'sample'")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature 必须是有限正数；Greedy 使用 strategy='greedy'")
    if top_k is not None:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k 必须为正整数或 None")
        if vocab_size is not None and top_k > vocab_size:
            raise ValueError(f"top_k={top_k} 超过词表大小 {vocab_size}")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int | None = None, strategy: str = "sample") -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError("logits 必须是非空 [B,V] Tensor")
    if not logits.is_floating_point():
        raise TypeError("logits 必须是浮点数")
    validate_sampling(temperature, top_k, strategy, logits.size(-1))
    # -inf 可以用来屏蔽候选，但每行至少有一个有限分数。
    if (torch.isnan(logits).any() or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError("logits 含 NaN、+Inf 或整行被屏蔽")
    if strategy == "greedy":
        return logits.argmax(dim=-1, keepdim=True)

    # 低精度转 FP32；先减最大值，减小温度缩放时溢出的风险。
    scores = logits.float()
    scores = (scores - scores.max(dim=-1, keepdim=True).values) / temperature
    if top_k is not None:
        values, indices = torch.topk(scores, k=top_k, dim=-1)
        # 按索引保留恰好 k 个候选，避免并列分数使候选数超过 k。
        scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, values)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)
