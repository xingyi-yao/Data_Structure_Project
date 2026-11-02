"""微调保持预训练模型结构和完整词表，重新初始化优化器。"""
from .train import build_parser, run_training


def main():
    run_training(build_parser(finetune=True).parse_args(), finetune=True)


if __name__ == "__main__":
    main()
