"""统一自回归入口；UI、测试与 Benchmark 均调用 generate。"""
from contextlib import contextmanager
from collections.abc import Mapping
import torch
from .sampling import sample_next_token, validate_sampling
from .kv_cache import validate_cache


def get_context_length(model, explicit=None) -> int:
    """兼容 config 字典/对象；也可显式传入 context_length。"""
    length = explicit
    if length is None:
        for source in (getattr(model, "config", None), model):
            for name in ("context_length", "block_size"):
                value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
                if value is not None:
                    length = value
                    break
            if length is not None:
                break
    if type(length) is not int or length < 1:
        raise ValueError("请提供正整数 context_length，或在 model.config 中设置该字段")
    return length


@contextmanager
def evaluating(model):
    """eval 关闭 Dropout；退出后恢复原状态，避免影响后续训练。"""
    modules = list(model.modules()) if isinstance(model, torch.nn.Module) else []
    states = [module.training for module in modules]
    try:
        if modules:
            model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in zip(modules, states):
            module.training = training


def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 100,
             temperature: float = 1.0, top_k: int | None = None,
             use_kv_cache: bool = False, *, context_length: int | None = None,
             strategy: str = "sample") -> torch.Tensor:
    """返回完整 prompt + 新 token：[B,T+max_new_tokens]。

    普通分支返回 (logits,loss)，缓存分支返回 (logits,present_kv)。
    未约定 EOS/PAD；每行 prompt 等长，固定生成指定数量，不自动补齐。
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("input_ids 必须是非空 [B,T] Tensor；空 prompt 请先给起始文本")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids 请使用 torch.long")
    if (input_ids < 0).any():
        raise ValueError("token ID 不可为负数")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须为非负整数")
    if type(use_kv_cache) is not bool:
        raise TypeError("use_kv_cache 必须为 bool")
    validate_sampling(temperature, top_k, strategy)
    if max_new_tokens == 0:
        return input_ids.clone()
    limit = get_context_length(model, context_length)
    idx, cache, cache_length, layers = input_ids, None, 0, None
    with evaluating(model):
        for _ in range(max_new_tokens):
            # Prefill：首轮输入整个 prompt；Decode：后续仅输入刚生成的 token。
            if use_kv_cache and cache is not None and cache_length < limit:
                idx_cond, previous = idx[:, -1:], cache
                expected_length = cache_length + 1
            else:
                idx_cond, previous = idx[:, -limit:], None
                expected_length = idx_cond.size(1)
                # 窗口满后重建缓存：位置重新编号，且深层状态依赖已移除的历史。
                # 只裁掉 K/V 头部不能保证与无缓存滑动窗口等价。
            if use_kv_cache:
                result = model(idx_cond, past_kv=previous, use_cache=True)
            else:
                result = model(idx_cond)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError("模型需返回 (logits,loss) 或缓存模式的 (logits,present_kv)")
            logits, second = result
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != idx_cond.shape:
                raise ValueError("logits 需要是 [B,T,V]，T 等于本轮输入长度")
            if logits.device != input_ids.device:
                raise ValueError("logits 和 input_ids 必须在同一 device")
            if use_kv_cache:
                cache_length = validate_cache(second, batch_size=idx.size(0),
                    expected_length=expected_length, expected_layers=layers, device=idx.device)
                cache, layers = second, len(second)
            # 位置 t 的输出预测 t+1，所以只取最后位置：[B,T,V] -> [B,V]。
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, strategy)
            idx = torch.cat((idx, next_token), dim=1)
    # cache 为局部变量；不同 prompt 调用之间不共享缓存。
    return idx
