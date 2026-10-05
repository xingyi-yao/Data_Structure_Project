"""B 的预训练脚本：普通文本 -> 训练 GPT -> 保存 checkpoint。

训练流程：
    文本数据 -> CharTokenizer 建词表 -> TextDataset 构造 (x, y)
    -> GPT 前向计算 loss -> 反向传播 -> 优化器更新参数
    -> 定期验证 -> 保存最好的模型
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.model.gpt import GPT
from src.training.dataset import TextDataset


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_device():
    """自动选择 GPU 或 CPU。有 CUDA 就用 GPU，否则用 CPU。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU（未检测到 CUDA）")
    return device


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """学习率调度：warmup 线性上升 + cosine decay 下降。

    GPT 标准做法：
    - 前 warmup_steps 步：学习率从 0 线性升到 max_lr（避免初期梯度太大）
    - 之后：cosine 曲线从 max_lr 慢慢降到 min_lr
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, dataset, eval_iters, batch_size, device):
    """在 val 集上算平均 loss。

    用 torch.no_grad() 是因为验证不需要反向传播，省显存省时间。
    """
    model.eval()  # 切换到评估模式（关闭 dropout）
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch("val", batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()  # 切回训练模式
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    torch.manual_seed(args.seed)

    # ---- 1. 数据 + 词表 ----
    print(f"\n加载数据: {args.data_path}")
    dataset = TextDataset.from_file(
        args.data_path,
        context_length=args.context_length,
        train_ratio=args.train_ratio,
    )
    vocab_size = dataset.tokenizer.vocab_size
    print(f"词表大小: {vocab_size}")
    print(f"训练 token 数: {len(dataset.train_ids)}")
    print(f"验证 token 数: {len(dataset.val_ids)}")

    # ---- 2. 模型 ----
    print(f"\n创建模型: n_embd={args.n_embd}, n_layer={args.n_layer}, "
          f"n_head={args.n_head}")
    model = GPT(
        vocab_size=vocab_size,
        context_length=args.context_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # ---- 3. 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # ---- 4. 保存目录 ----
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 tokenizer（推理时必须用同一份词表）
    tokenizer_path = ckpt_dir / "tokenizer.json"
    dataset.tokenizer.save(str(tokenizer_path))
    print(f"\n词表已保存: {tokenizer_path}")

    # ---- 5. 训练循环 ----
    print(f"\n开始训练，共 {args.max_steps} 步")
    train_losses = []   # [(step, loss), ...]
    val_losses = []     # [(step, loss), ...]
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(args.max_steps):
        # 设置学习率
        lr = get_lr(step, args.warmup_steps, args.max_steps,
                    args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 标准训练四步
        x, y = dataset.get_batch("train", args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        # 梯度裁剪：把所有参数的梯度范数限制在 grad_clip 以内
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # 记录训练 loss
        if step % args.log_interval == 0:
            train_losses.append((step, loss.item()))
            elapsed = time.time() - t0
            print(f"step {step:6d} | lr {lr:.2e} | "
                  f"train_loss {loss.item():.4f} | {elapsed:.1f}s")

        # 验证 + 保存最好的 checkpoint
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, dataset, args.eval_iters,
                                args.batch_size, device)
            val_losses.append((step, val_loss))
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "vocab_size": vocab_size,
                        "context_length": args.context_length,
                        "n_embd": args.n_embd,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "dropout": args.dropout,
                    },
                }
                best_path = ckpt_dir / "pretrain_best.pt"
                torch.save(checkpoint, str(best_path))
                print(f"  -> 保存最好的模型: {best_path} (val_loss={val_loss:.4f})")

    # ---- 6. 保存最后的模型 ----
    last_path = ckpt_dir / "pretrain_last.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": args.max_steps,
        "best_val_loss": best_val_loss,
        "config": {
            "vocab_size": vocab_size,
            "context_length": args.context_length,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "dropout": args.dropout,
        },
    }, str(last_path))
    print(f"\n保存最终模型: {last_path}")

    # ---- 7. 保存训练日志 ----
    log_path = out_dir / "training_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"训练步数: {args.max_steps}\n")
        f.write(f"词表大小: {vocab_size}\n")
        f.write(f"模型参数量: {n_params:,}\n")
        f.write(f"最好 val_loss: {best_val_loss:.4f}\n")
        f.write(f"\n训练 loss:\n")
        for step, loss in train_losses:
            f.write(f"step {step}: {loss:.4f}\n")
        f.write(f"\n验证 loss:\n")
        for step, loss in val_losses:
            f.write(f"step {step}: {loss:.4f}\n")
    print(f"训练日志已保存: {log_path}")

    # ---- 8. 画 loss 曲线 ----
    try:
        import matplotlib
        matplotlib.use("Agg")  # 不弹窗，直接存文件
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        if train_losses:
            steps = [s for s, _ in train_losses]
            losses = [l for _, l in train_losses]
            ax.plot(steps, losses, label="train loss", alpha=0.7)
        if val_losses:
            steps = [s for s, _ in val_losses]
            losses = [l for _, l in val_losses]
            ax.plot(steps, losses, label="val loss", linewidth=2)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        curve_path = out_dir / "loss_curve.png"
        fig.savefig(str(curve_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"loss 曲线已保存: {curve_path}")
    except ImportError:
        print("matplotlib 未安装，跳过 loss 曲线绘制")

    print(f"\n训练完成！最好 val_loss = {best_val_loss:.4f}")
    return best_val_loss


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPT 预训练脚本")
    parser.add_argument("--data_path", type=str,
                        default="data/raw/pretrain.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs/training")

    # 模型配置
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    # 日志配置
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

