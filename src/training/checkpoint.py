"""B 的 checkpoint 模块：封装模型的保存与加载。

为什么需要这个文件：
- train.py 训练完会保存模型
- finetune.py 需要加载预训练模型继续训
- evaluate.py 需要加载模型做评估
如果每个文件都写一遍 torch.load，会重复且容易出错。
所以统一封装在这里。

保存的内容（一个字典）：
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
    }
}
"""

from pathlib import Path

import torch

from src.model.gpt import GPT


def save_checkpoint(model, optimizer, step, best_val_loss, config, path):
    """保存 checkpoint。

    Args:
        model: GPT 模型
        optimizer: 优化器
        step: 当前训练步数
        best_val_loss: 最好的验证 loss
        config: dict，模型配置（vocab_size, n_embd, n_layer, ...）
        path: 保存路径
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))


def load_checkpoint(path, device):
    """加载 checkpoint，返回模型和训练状态。

    用于"继续训练"的场景：需要重建模型 + 优化器 + 恢复步数。

    Returns:
        model: 加载好权重的 GPT 模型
        optimizer: 恢复状态的优化器
        step: 训练到的步数
        best_val_loss: 最好的验证 loss
        config: 模型配置
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

    return model, optimizer, step, best_val_loss, config


def load_model_only(path, device):
    """只加载模型权重（不需要优化器）。

    用于"推理/评估"的场景：只需要模型，不需要继续训练。

    Returns:
        model: 加载好权重的 GPT 模型
        config: 模型配置
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
    return model, config


if __name__ == "__main__":
    # 自测：保存一个随机模型，再加载回来，验证权重一致。
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

    # 保存
    test_path = "test_checkpoint.pt"
    save_checkpoint(model, optimizer, step=100, best_val_loss=2.5,
                    config=config, path=test_path)
    print(f"已保存: {test_path}")

    # 加载
    loaded_model, loaded_opt, step, best_loss, loaded_config = \
        load_checkpoint(test_path, device="cpu")
    print(f"加载成功: step={step}, best_val_loss={best_loss}")
    print(f"config 一致: {loaded_config == config}")

    # 验证权重一致
    for (name1, p1), (name2, p2) in zip(model.named_parameters(),
                                         loaded_model.named_parameters()):
        assert name1 == name2
        assert torch.equal(p1, p2), f"权重不一致: {name1}"
    print("所有权重一致，checkpoint.py 自测 OK")

    # 清理
    Path(test_path).unlink(missing_ok=True)
