# 以下是仓库结构（暂定）
```
small-gpt/
│
├── README.md                       # 项目总说明，三人共同维护
├── requirements.txt               # Python 依赖
├── .gitignore
│
├── data/                           # 【B 主负责】数据
│   ├── README.md                   # 数据来源、清洗、去重、划分说明
│   ├── raw/
│   │   ├── pretrain.txt            # 预训练数据（莎士比亚戏剧）
│   │   └── finetune.txt            # 微调数据（莎士比亚十四行诗）
│   │
│   └── processed/                  # 划分后的训练/验证/测试集
│       ├── pretrain/
│       │   ├── train.txt
│       │   ├── val.txt
│       │   └── test.txt
│       └── finetune/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
│
├── configs/                        # 【B】训练超参数配置
│   ├── pretrain.json
│   └── finetune.json
│
├── src/
│   │
│   ├── tokenizer/                  # 【C】
│   │   ├── __init__.py
│   │   ├── char_tokenizer.py       # 第一版 Character Tokenizer
│   │   └── bpe_tokenizer.py        # 可选：自己实现 BPE
│   │
│   ├── model/                      # 【A】
│   │   ├── __init__.py
│   │   ├── attention.py            # Self-Attention / Multi-Head Attention
│   │   ├── feedforward.py          # FFN
│   │   ├── block.py                # Transformer Block
│   │   └── gpt.py                  # 完整 GPT-style Model
│   │
│   ├── training/                   # 【B】
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset / get_batch
│   │   ├── train.py                # Pre-training
│   │   ├── finetune.py             # Fine-Tuning
│   │   ├── evaluate.py             # train / val loss
│   │   └── checkpoint.py           # checkpoint 保存和恢复
│   │
│   ├── inference/                  # 【C】
│   │   ├── __init__.py
│   │   ├── generate.py             # 自回归文本生成
│   │   ├── sampling.py             # greedy / temperature / top-k
│   │   └── kv_cache.py             # KV Cache 相关逻辑
│   │
│   ├── optimization/               # 【C 主负责，A 配合】
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── kv_cache_benchmark.py
│   │   └── attention_benchmark.py
│   │
├── ui/                             # ⭐【C 新增负责】
│   ├── app.py                      # UI 主程序
│   ├── model_service.py            # UI 与模型之间的接口
│   ├── config.py                   # UI 默认参数
│   │
│   ├── pages/                      # 如果以后做多页面
│   │   ├── generation.py           # 文本生成页面
│   │   ├── finetune_demo.py        # Fine-Tuning 前后对比
│   │   └── benchmark.py            # 性能测试展示
│   │
│   └── assets/
│       └── README.md
│
├── tests/                          # 【C 主负责测试】
│   ├── test_attention.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_generation.py
│   └── test_kv_cache.py
│
├── checkpoints/                    # 【B 生成】
│   ├── tokenizer.json              # 词表（预训练+微调共用）
│   ├── pretrain_best.pt
│   ├── pretrain_last.pt
│   ├── finetune_best.pt
│   └── finetune_last.pt
│
├── outputs/
│   ├── training/                   # 【B】
│   │   ├── loss_curve.png
│   │   └── training_log.txt
│   │
│   ├── generation/                 # 【C】
│   │   └── generation_samples.txt
│   │
│   ├── finetune/                   # 【B】
│   │   ├── before_after.txt
│   │   ├── finetune_log.txt
│   │   ├── loss_curve.png
│   │   ├── specialized_evidence.txt
│   │   └── failure_cases.txt
│   │
│   └── benchmark/                  # 【C】
│       ├── benchmark_results.txt
│       └── benchmark_curve.png
│
├── scripts/                        # 方便最终答辩直接运行
│   ├── train.sh
│   ├── finetune.sh
│   ├── generate.sh
│   └── run_ui.sh
│
└── docs/
    ├── architecture.md             # 【A】
    ├── training.md                 # 【B】
    ├── inference.md                # 【C】
    └── ui.md                       # 【C】
```
