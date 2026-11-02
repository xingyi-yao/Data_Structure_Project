from config import PROJECT_ROOT

# 优先使用 checkpoint 内嵌词表；外部词表只用于兼容旧 checkpoint。
MODEL_SPECS = {
    "pretrain": {"checkpoint": PROJECT_ROOT / "checkpoints/pretrain_best.pt", "tokenizer": None},
    "finetune": {"checkpoint": PROJECT_ROOT / "checkpoints/finetune_best.pt", "tokenizer": None},
}
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_SEED = 42
DEFAULT_BENCHMARK_LENGTHS = "50,100,200,500"
