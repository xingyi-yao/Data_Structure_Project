"""预训练命令：python -m src.training.train。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from config import PROJECT_ROOT, ModelConfig, TrainConfig
from src.model.gpt import GPT
from src.tokenizer import CharTokenizer, BPETokenizer
from src.utils import get_device, set_seed
from src.utils.logger import get_logger
from .dataset import prepare_data, TextDataset
from .evaluate import estimate_loss
from .checkpoint import save_checkpoint, load_checkpoint


def build_parser(finetune=False):
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Small GPT 微调" if finetune else "Small GPT 预训练")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / ("data/raw/finetune.txt" if finetune else "data/raw/pretrain.txt"))
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / ("data/processed/finetune" if finetune else "data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / ("outputs/finetune" if finetune else "outputs/training"))
    parser.add_argument("--resume", type=Path)
    if finetune:
        parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/pretrain_best.pt")
    for name in ("batch_size", "max_steps", "eval_interval", "eval_batches", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "grad_clip", "val_fraction"):
        default = 1e-4 if finetune and name == "learning_rate" else getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--device", default="auto")
    if not finetune:
        parser.add_argument("--context-length", type=int, default=128)
        parser.add_argument("--n-embd", type=int, default=64)
        parser.add_argument("--n-layer", type=int, default=2)
        parser.add_argument("--n-head", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
        parser.add_argument("--vocab-size", type=int, default=512)
    return parser


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots()
    for key in ("train", "val"):
        axes.plot([row["step"] for row in history], [row[key] for row in history], label=key)
    axes.set(xlabel="Training step", ylabel="Cross entropy loss")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run_training(args, finetune=False):
    for name in ("batch_size", "max_steps", "eval_interval", "eval_batches"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} 必须大于零")
    for name in ("learning_rate", "grad_clip", "weight_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"{name} 不合法")
    set_seed(args.seed)
    device = get_device(args.device)
    train_text, val_text = prepare_data(args.data, args.processed_dir, args.val_fraction, args.val_data)
    signature = hashlib.sha256(json.dumps([train_text, val_text], ensure_ascii=False).encode("utf-8")).hexdigest()
    state = {}
    source = args.resume or (args.checkpoint if finetune else None)
    if source:
        model, tokenizer, state = load_checkpoint(source, device)
    else:
        tokenizer = CharTokenizer(train_text) if args.tokenizer == "char" else BPETokenizer(train_text, args.vocab_size)
        config = ModelConfig(tokenizer.vocab_size, args.context_length, args.n_embd,
                             args.n_layer, args.dropout, args.n_head)
        model = GPT(**config.to_dict()).to(device)
    # 词表只从训练集学习；验证集/微调集遇到未知字符时明确报错。
    datasets = {name: TextDataset(tokenizer.encode(text), model.context_length)
                for name, text in (("train", train_text), ("val", val_text))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start, best, history = 0, float("inf"), []
    if args.resume:
        if "optimizer_state_dict" not in state:
            raise ValueError("断点缺少 optimizer_state_dict，无法续训")
        if state.get("data_signature") not in (None, signature):
            raise ValueError("续训数据与 checkpoint 不一致；换语料请使用微调命令")
        previous_task = state.get("training_config", {}).get("checkpoint")
        if state.get("training_config") and bool(previous_task) != finetune:
            raise ValueError("续训必须使用原任务入口；预训练转微调请使用 --checkpoint")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start = state["training_step"]
        best = state.get("best_validation_loss", state["validation_loss"])
        history = list(state.get("history", []))
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in state:
            if len(state["cuda_rng_states"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    if start >= args.max_steps:
        raise ValueError("max_steps 是总步数，必须大于 checkpoint 中的 training_step")
    prefix = "finetune" if finetune else "pretrain"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # 微调词表随权重内嵌，不覆盖可能属于其他模型的共享词表。
    tokenizer.save(args.checkpoint_dir / f"{prefix}_tokenizer.json")
    if not finetune and not args.resume:
        tokenizer.save(args.checkpoint_dir / "tokenizer.json")
    logger = get_logger(args.output_dir / "training_log.txt")
    logger.info("task=%s device=%s vocab=%s start=%s total=%s", prefix, device, tokenizer.vocab_size, start, args.max_steps)
    training_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    model.train()
    for step in range(start + 1, args.max_steps + 1):
        x, y = datasets["train"].get_batch(args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"第 {step} 步出现非有限 loss，停止保存异常权重")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.max_steps:
            losses = estimate_loss(model, datasets, batch_size=args.batch_size,
                                   eval_batches=args.eval_batches, device=device)
            if not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError("评估损失非有限，停止保存异常权重")
            history.append(dict(step=step, **losses))
            improved = losses["val"] < best
            best = min(best, losses["val"])
            metadata = dict(training_step=step, validation_loss=losses["val"],
                            best_validation_loss=best, history=history,
                            training_config=training_config, data_signature=signature)
            save_checkpoint(args.checkpoint_dir / f"{prefix}_last.pt", model, tokenizer, optimizer, **metadata)
            if improved:
                save_checkpoint(args.checkpoint_dir / f"{prefix}_best.pt", model, tokenizer, optimizer, **metadata)
            logger.info("step=%d train=%.6f val=%.6f best=%.6f", step, losses["train"], losses["val"], best)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, args.output_dir / "loss_curve.png")
    return model, tokenizer, history


def main():
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
