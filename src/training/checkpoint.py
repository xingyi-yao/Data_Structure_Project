"""B 的 checkpoint 模块：封装模型的保存与加载。

为什么需要这个文件：
- train.py 训练完会保存模型
- finetune.py 需要加载预训练模型继续训
- evaluate.py 需要加载模型做评估
如果每个文件都写一遍 torch.load，会重复且容易出错。
所以统一封装在这里。

保存的内容（一个字典，符合分工文件 7.3 节要求）：
{
    "model_state_dict": ...,      # 模型权重
    "optimizer_state_dict": ...,  # 优化器状态（继续训练时需要）
    "step": ...,                  # 训练到第几步
    "best_val_loss": ...,         # 最好的验证 loss
    "config": {                   # 模型配置（重建模型时需要）
        "vocab_size": ...,
        "context_length": ...,
        "n_embd": ...,
        "n_layer": ...,
        "n_head": ...,
        "dropout": ...,
    },
    # 分工 7.3 节新增字段：
    "stage": "pretrain" / "finetune",   # 训练阶段
    "tokenizer_hash": ...,              # 词表哈希，加载时验证一致性
    "data_version": ...,                # 数据文件哈希，可追溯
    "parent_checkpoint": ...,            # 微调时记录预训练 checkpoint 路径
    "rng_states": ...,                  # 随机数状态，保证可复现
}
"""

import hashlib
from pathlib import Path

import torch

from src.model.gpt import GPT


# ---------------------------------------------------------------------------
# 辅助函数：计算哈希、捕获 RNG 状态
# ---------------------------------------------------------------------------

def compute_tokenizer_hash(tokenizer):
    """计算 tokenizer 词表的 MD5 哈希。

    用于加载 checkpoint 时验证：当前 tokenizer 是否和保存时一致。
    不一致说明词表变了，模型权重对不上。
    """
    if tokenizer is None:
        return None
    # 把词表按字符排序后拼接，再哈希
    # 这样保证同样的词表（不管顺序）算出同样的哈希
    chars = sorted(tokenizer.stoi.keys())
    content = "".join(chars)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def compute_data_version(data_path):
    """计算数据文件的 MD5 哈希。

    用于追溯：这个 checkpoint 是用哪份数据训出来的。
    """
    if data_path is None:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_rng_states():
    """捕获当前所有随机数生成器的状态。

    包括 CPU 和（如果有）CUDA 的 RNG 状态。
    保存这个可以让训练完全可复现（配合 seed）。
    """
    states = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state()
    return states


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, best_val_loss, config, path,
                    stage="pretrain", tokenizer=None, data_path=None,
                    parent_checkpoint=None, train_config=None):
    """保存 checkpoint。

    Args:
        model: GPT 模型
        optimizer: 优化器
        step: 当前训练步数
        best_val_loss: 最好的验证 loss
        config: dict，模型配置（vocab_size, n_embd, n_layer, ...）
        path: 保存路径
        stage: "pretrain" 或 "finetune"（训练阶段）
        tokenizer: tokenizer 对象（计算 tokenizer_hash 用）
        data_path: 数据文件路径（计算 data_version 用）
        parent_checkpoint: 父 checkpoint 路径（微调时是预训练 checkpoint）
        train_config: dict，训练配置（batch_size, learning_rate, max_steps 等）
    """
    checkpoint = {
        # 原有字段
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": config,
        "train_config": train_config,
        # 兼容旧代码: 保留 config 字段(等同于 model_config)
        "config": config,
        # 分工 7.3 节新增字段
        "stage": stage,
        "tokenizer_hash": compute_tokenizer_hash(tokenizer),
        "data_version": compute_data_version(data_path),
        "parent_checkpoint": parent_checkpoint,
        "rng_states": capture_rng_states(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Windows 下直接覆盖可能被锁定，先存临时文件再替换
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, str(tmp_path))
    if path.exists():
        path.unlink()
    tmp_path.rename(path)


def load_checkpoint(path, device, tokenizer=None):
    """加载 checkpoint，返回模型和训练状态。

    用于"继续训练"的场景：需要重建模型 + 优化器 + 恢复步数。

    Args:
        path: checkpoint 路径
        device: 加载到哪个设备
        tokenizer: 可选，传入后会校验 tokenizer_hash 是否一致

    Returns:
        model: 加载好权重的 GPT 模型
        optimizer: 恢复状态的优化器
        step: 训练到的步数
        best_val_loss: 最好的验证 loss
        config: 模型配置
        meta: dict，包含 stage / tokenizer_hash / data_version /
              parent_checkpoint / rng_states（兼容旧版 checkpoint）
    """
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config = checkpoint["config"]

    # 用保存的配置重建模型
    model = GPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        dropout=config["dropout"],
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 重建优化器并加载状态
    optimizer = None
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # 新字段（兼容旧 checkpoint：没有就给 None）
    meta = {
        "stage": checkpoint.get("stage"),
        "tokenizer_hash": checkpoint.get("tokenizer_hash"),
        "data_version": checkpoint.get("data_version"),
        "parent_checkpoint": checkpoint.get("parent_checkpoint"),
        "rng_states": checkpoint.get("rng_states"),
    }

    # 校验 tokenizer 一致性（如果传入了 tokenizer）
    if tokenizer is not None and meta["tokenizer_hash"] is not None:
        current_hash = compute_tokenizer_hash(tokenizer)
        if current_hash != meta["tokenizer_hash"]:
            print(f"⚠️  警告：tokenizer 哈希不匹配！")
            print(f"   checkpoint 保存时: {meta['tokenizer_hash']}")
            print(f"   当前 tokenizer:   {current_hash}")
            print(f"   词表可能已变化，模型权重可能对不上。")

    return model, optimizer, step, best_val_loss, config, meta


def load_model_only(path, device):
    """只加载模型权重（不需要优化器）。

    用于"推理/评估"的场景：只需要模型，不需要继续训练。

    Returns:
        model: 加载好权重的 GPT 模型
        config: 模型配置
        meta: dict，包含 stage 等元信息（兼容旧版）
    """
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 评估模式

    meta = {
        "stage": checkpoint.get("stage"),
        "tokenizer_hash": checkpoint.get("tokenizer_hash"),
        "data_version": checkpoint.get("data_version"),
        "parent_checkpoint": checkpoint.get("parent_checkpoint"),
        "rng_states": checkpoint.get("rng_states"),
    }
    return model, config, meta


if __name__ == "__main__":
    # 自测：保存一个随机模型，再加载回来，验证权重一致 + 新字段正确。
    torch.manual_seed(42)

    config = {
        "vocab_size": 10,
        "context_length": 8,
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 2,
        "dropout": 0.0,
    }

    # 创建一个随机模型
    model = GPT(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 假装有 tokenizer 和数据
    class FakeTokenizer:
        def __init__(self):
            self.stoi = {"a": 0, "b": 1, "c": 2}

    tok = FakeTokenizer()

    # 保存
    test_path = "test_checkpoint.pt"
    save_checkpoint(model, optimizer, step=100, best_val_loss=2.5,
                    config=config, path=test_path,
                    stage="pretrain", tokenizer=tok,
                    data_path=None, parent_checkpoint=None)
    print(f"已保存: {test_path}")

    # 加载
    loaded_model, loaded_opt, step, best_loss, loaded_config, meta = \
        load_checkpoint(test_path, device="cpu", tokenizer=tok)
    print(f"加载成功: step={step}, best_val_loss={best_loss}")
    print(f"config 一致: {loaded_config == config}")
    print(f"meta: stage={meta['stage']}, tokenizer_hash={meta['tokenizer_hash'][:8]}, "
          f"rng_states={'cpu' in (meta['rng_states'] or {})}")

    # 验证权重一致
    for (name1, p1), (name2, p2) in zip(model.named_parameters(),
                                         loaded_model.named_parameters()):
        assert name1 == name2
        assert torch.equal(p1, p2), f"权重不一致: {name1}"
    print("所有权重一致，checkpoint.py 自测 OK")

    # 清理
    Path(test_path).unlink(missing_ok=True)

