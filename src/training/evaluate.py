"""B 的评估脚本：对比微调前后的生成效果。

流程：
1. 加载预训练模型（微调前）和微调模型（微调后）
2. 用相同的 prompt 分别生成文本
3. 把对比结果保存到 outputs/finetune/before_after.txt
"""

import argparse
from pathlib import Path

import torch

from src.tokenizer.char_tokenizer import CharTokenizer
from src.training.checkpoint import load_model_only
from src.inference.generate import generate


def generate_text(model, tokenizer, prompt, max_new_tokens, device, temperature=0.7, top_k=50):
    """用模型生成文本。

    Args:
        model: GPT 模型
        tokenizer: CharTokenizer
        prompt: 字符串
        max_new_tokens: 生成多少个新 token
        device: cpu/cuda

    Returns:
        生成的字符串
    """
    # prompt 字符串 -> token IDs -> [1, T] Tensor
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    # 调用 C 的 generate
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_kv_cache=False,   # A 还没接 KV Cache，必须 False
        context_length=model.config["context_length"]
        if hasattr(model, "config") else None,
    )

    # token IDs -> 字符串
    return tokenizer.decode(output_ids[0].tolist())


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载词表（预训练和微调共用同一份）
    tokenizer = CharTokenizer.load(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # 2. 加载预训练模型（微调前）
    print(f"\nLoading pretrain model: {args.pretrain_path}")
    pretrain_model, pretrain_config, _ = load_model_only(args.pretrain_path, device)
    print(f"Pretrain model loaded")

    # 3. 加载微调模型（微调后）
    print(f"Loading finetune model: {args.finetune_path}")
    finetune_model, finetune_config, _ = load_model_only(args.finetune_path, device)
    print(f"Finetune model loaded")

    # 4. 读取 prompts
    prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
    print(f"\n{len(prompts)} prompts")

    # 5. 对比生成
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i + 1}: {prompt!r} ---")

        before = generate_text(pretrain_model, tokenizer, prompt,
                               args.max_new_tokens, device,
                               temperature=args.temperature, top_k=args.top_k)
        after = generate_text(finetune_model, tokenizer, prompt,
                              args.max_new_tokens, device,
                              temperature=args.temperature, top_k=args.top_k)

        print(f"[Before] {before}")
        print(f"[After] {after}")

        results.append((prompt, before, after))

    # 6. 保存对比结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "before_after.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Before/After Finetune Comparison\n")
        f.write("=" * 60 + "\n\n")
        for prompt, before, after in results:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"[Before] {before}\n")
            f.write(f"[After] {after}\n")
            f.write("-" * 60 + "\n")

    print(f"\nComparison saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Before/after finetune comparison")
    parser.add_argument("--pretrain_path", type=str,
                        default="checkpoints/pretrain_best.pt")
    parser.add_argument("--finetune_path", type=str,
                        default="checkpoints/finetune_best.pt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="checkpoints/tokenizer.json")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/finetune")
    parser.add_argument("--prompts", type=str,
                        default="To be|Shall I|When in",
                        help="用 | 分隔多个 prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
