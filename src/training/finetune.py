"""B 的诗歌微调脚本：加载预训练模型 -> 用诗歌继续训练 -> 保存微调模型。

与 train.py 的区别：
- train.py: 随机初始化模型，用普通文本预训练
- finetune.py: 加载 pretrain_best.pt，用诗歌数据微调

微调要点：
- 复用预训练的词表（tokenizer.json），不能新建
- 学习率比预训练小很多（避免破坏已学知识）
- 全参数微调（不冻结任何层）
"""

import argparse
import math
import time
from pathlib import Path

import torch

from src.model.gpt import GPT
from src.tokenizer.char_tokenizer import CharTokenizer
from src.training.dataset import TextDataset
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.train import get_device, get_lr, evaluate


def finetune(args):
    device = get_device()
    torch.manual_seed(args.seed)

    # ---- 1. 加载预训练模型 ----
    print(f"\n加载预训练模型: {args.pretrain_path}")
    model, _, _, _, config, _ = load_checkpoint(args.pretrain_path, device)
    print(f"预训练模型加载成功: vocab_size={config['vocab_size']}, "
          f"n_layer={config['n_layer']}")

    # ---- 2. 加载预训练词表（必须复用，不能新建）----
    tokenizer = CharTokenizer.load(args.tokenizer_path)
    print(f"词表加载成功: vocab_size={tokenizer.vocab_size}")

    # ---- 3. 加载诗歌数据 ----
    print(f"\n加载微调数据: {args.data_path}")
    text = Path(args.data_path).read_text(encoding="utf-8")
    dataset = TextDataset.from_text(
        text,
        context_length=config["context_length"],
        train_ratio=args.train_ratio,
        tokenizer=tokenizer,   # 复用预训练词表
        split_by_line=True,    # 按整首诗划分 train/val
    )
    print(f"训练 token 数: {len(dataset.train_ids)}")
    print(f"验证 token 数: {len(dataset.val_ids)}")

    # ---- 4. 优化器（新建，因为微调学习率不同）----
    # 注意：不加载预训练的 optimizer，因为学习率和训练状态不同
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # ---- 5. 保存目录 ----
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 6. 微调循环 ----
    print(f"\n开始微调，共 {args.max_steps} 步")
    print(f"学习率: {args.learning_rate}（比预训练小，避免破坏已学知识）")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(args.max_steps):
        # 学习率调度
        lr = get_lr(step, args.warmup_steps, args.max_steps,
                    args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 标准训练四步
        x, y = dataset.get_batch("train", args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # 记录 loss
        if step % args.log_interval == 0:
            train_losses.append((step, loss.item()))
            elapsed = time.time() - t0
            print(f"step {step:6d} | lr {lr:.2e} | "
                  f"train_loss {loss.item():.4f} | {elapsed:.1f}s")

        # 验证 + 保存最好的微调模型
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, dataset, args.eval_iters,
                                args.batch_size, device)
            val_losses.append((step, val_loss))
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    config=config,
                    path=ckpt_dir / "finetune_best.pt",
                    stage="finetune",
                    tokenizer=tokenizer,
                    data_path=args.data_path,
                    parent_checkpoint=args.pretrain_path,
                )
                print(f"  -> 保存最好的微调模型: finetune_best.pt "
                      f"(val_loss={val_loss:.4f})")

    # ---- 6.5 测试集评估 ----
    test_loss = evaluate(model, dataset, args.eval_iters, args.batch_size,
                         device, split="test")
    print(f"\n测试集 loss: {test_loss:.4f}")

    # ---- 7. 保存最后的微调模型 ----
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=args.max_steps,
        best_val_loss=best_val_loss,
        config=config,
        path=ckpt_dir / "finetune_last.pt",
        stage="finetune",
        tokenizer=tokenizer,
        data_path=args.data_path,
        parent_checkpoint=args.pretrain_path,
    )
    print(f"\n保存最终微调模型: finetune_last.pt")

    # ---- 8. 保存微调日志 ----
    log_path = out_dir / "finetune_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"微调步数: {args.max_steps}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"最好 val_loss: {best_val_loss:.4f}\n")
        f.write(f"测试集 loss: {test_loss:.4f}\n")
        f.write(f"\n训练 loss:\n")
        for step, loss in train_losses:
            f.write(f"step {step}: {loss:.4f}\n")
        f.write(f"\n验证 loss:\n")
        for step, loss in val_losses:
            f.write(f"step {step}: {loss:.4f}\n")
    print(f"微调日志已保存: {log_path}")

    print(f"\n微调完成！最好 val_loss = {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="GPT 诗歌微调脚本")
    parser.add_argument("--pretrain_path", type=str,
                        default="checkpoints/pretrain_best.pt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="checkpoints/tokenizer.json")
    parser.add_argument("--data_path", type=str,
                        default="data/raw/finetune.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")

    # 微调配置（学习率比预训练小很多）
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    # 日志配置
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
