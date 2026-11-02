#### 简介：
三个大二学生的数据结构作业，从零开始、不使用 transformers 等高级库，仅用 Python 和 PyTorch 基础张量操作实现一个小型 GPT 风格语言模型，核心包括字符级分词器、token 与位置嵌入、由因果自注意力和前馈网络组成的 Transformer 模块（含 LayerNorm 与残差连接）、因果掩码、交叉熵损失、反向传播与 AdamW 参数更新，以及自回归文本生成；在基本要求之外，我们还计划扩展多头注意力、多层堆叠、GPT-2 风格架构、特定领域微调，并通过 KV 缓存优化自注意力计算效率；项目由三名成员分工协作，A 负责注意力与 GPT 架构，B 负责数据、预训练、检查点与微调，C 负责分词器、生成、测试与性能基准，三人共同完成最小模型跑通、KV 缓存集成以及最终系统验证，预期交付损失曲线、生成样本、微调对比、缓存加速基准、源代码、检查点和 README，本质上是在动手实现现代大语言模型背后的基础数据结构与算法。


# 以下是仓库结构（暂定）
```
small-gpt/
│
├── README.md                       # 项目总说明，三人共同维护
├── requirements.txt               # Python 依赖
├── config.py                       # 全局模型/训练配置
├── .gitignore
│
├── data/                           # 【B 主负责】数据
│   ├── raw/
│   │   ├── pretrain.txt
│   │   └── finetune.txt
│   │
│   └── processed/
│       ├── train.txt
│       └── val.txt
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
│   └── utils/                      # 【共同】
│       ├── __init__.py
│       ├── device.py
│       ├── seed.py
│       └── logger.py
│
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
│   ├── pretrain_best.pt
│   ├── pretrain_last.pt
│   └── finetune_best.pt
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
│   │   └── before_after.txt
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
