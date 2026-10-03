"""KV Cache 数据约定与检查；Attention 的内部计算由 A 实现。

TODO(A)：模型需要提供以下两个 forward 分支：
    logits, loss = model(idx, targets=None)
    logits, present_kv = model(idx, past_kv=None, use_cache=True)

past_kv/present_kv = [(K_layer0, V_layer0), (K_layer1, V_layer1), ...]
每个 K/V 为 [B, num_heads, T_cache, head_dim]，沿 dim=-2 拼接。
A 应让位置编码从 past_length 开始，因果 mask 也包含此偏移。
首次 past_kv=None 时从位置 0 开始；每层保存自己的 K/V。
"""
import torch

LayerKV = tuple[torch.Tensor, torch.Tensor]
KVCache = list[LayerKV] | tuple[LayerKV, ...]


def validate_cache(cache: KVCache, *, batch_size=None, expected_length=None,
                   expected_layers=None, device=None) -> int:
    """检查结构，返回缓存长度；不复制 GPU Tensor。"""
    if not isinstance(cache, (list, tuple)) or not cache:
        raise ValueError("present_kv 必须是非空的逐层 (K,V) 列表")
    if expected_layers is not None and len(cache) != expected_layers:
        raise ValueError("缓存层数发生变化")
    length = None
    for layer, pair in enumerate(cache):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"第 {layer} 层需要 (K,V)")
        key, value = pair
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("K/V 必须是 Tensor")
        if key.ndim != 4 or key.shape != value.shape or min(key.shape) < 1:
            raise ValueError(f"第 {layer} 层 K/V 必须同形且为 [B,H,T,D]")
        if key.device != value.device or key.dtype != value.dtype or not key.is_floating_point():
            raise ValueError("K/V 必须同 device、同浮点 dtype")
        if device is not None and key.device != torch.device(device):
            raise ValueError("缓存和输入不在同一 device")
        if batch_size is not None and key.size(0) != batch_size:
            raise ValueError("缓存 batch_size 与输入不符")
        if length is None:
            length = key.size(-2)
        if key.size(-2) != length:
            raise ValueError("所有层的缓存长度必须相同")
    if expected_length is not None and length != expected_length:
        raise ValueError(f"缓存应包含 {expected_length} 个位置，实际为 {length}")
    return length


def append_layer_kv(past: LayerKV | None, key: torch.Tensor,
                    value: torch.Tensor) -> LayerKV:
    """A 可在 Attention 内调用：沿序列维拼接。"""
    validate_cache([(key, value)])
    if past is None:
        return key, value
    validate_cache([past], batch_size=key.size(0), device=key.device)
    old_key, old_value = past
    if old_key.shape[1::2] != key.shape[1::2] or old_key.dtype != key.dtype:
        raise ValueError("新旧 K/V 的 head 数、head_dim、dtype 必须一致")
    return torch.cat((old_key, key), dim=-2), torch.cat((old_value, value), dim=-2)


def cache_nbytes(cache: KVCache) -> int:
    """Tensor 逻辑字节数；不等于 GPU 分配的峰值。"""
    validate_cache(cache)
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)
