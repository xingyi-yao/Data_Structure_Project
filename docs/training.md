# 训练文档

本文档记录 B 负责的训练模块的完整流程、参数配置和实验结果。

## 1. 训练概述

本项目采用**两阶段训练**策略:

1. **预训练 (Pretrain)**: 在莎士比亚戏剧全集上训练, 学习基础英语语言规律
2. **微调 (Finetune)**: 在莎士比亚十四行诗上继续训练, 学习诗歌风格

模型架构为字符级 GPT (Character-level Transformer), 使用 C 提供的 `CharTokenizer` 和 A 提供的 `GPT` 模型。

## 2. 数据

### 2.1 预训练数据

- **来源**: Project Gutenberg #100, The Complete Works of William Shakespeare
- **内容**: 全部戏剧 (已移除十四行诗部分, 保证交叉去重)
- **大小**: 5.1 MB (约 526 万字符)
- **文件**: `data/raw/pretrain.txt`
- **划分**: 90% train / 5% val / 5% test (按 token 位置连续切分)

### 2.2 微调数据

- **来源**: 莎士比亚十四行诗 (共 154 首)
- **大小**: 97 KB
- **文件**: `data/raw/finetune.txt`
- **划分**: 90% train / 5% val / 5% test (按整首诗段落切分, 不破坏单首诗完整性)

### 2.3 词表

- **建立方式**: 预训练 + 微调两阶段训练集字符并集
- **词表大小**: 100 个字符
- **保存**: `checkpoints/tokenizer.json`
- **说明**: 预训练和微调使用同一份词表, 保证模型输出维度一致

## 3. 模型配置

| 参数 | 预训练 | 微调 |
|------|--------|------|
| n_embd | 512 | (继承) |
| n_layer | 10 | (继承) |
| n_head | 8 | (继承) |
| context_length | 256 | 256 |
| dropout | 0.1 | 0.1 |
| vocab_size | 100 | 100 |

模型参数量约 20M。

## 4. 训练配置

### 4.1 预训练

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| learning_rate | 3e-4 |
| min_lr | 3e-5 |
| max_steps | 20000 |
| warmup_steps | 100 |
| grad_clip | 1.0 |
| weight_decay | 0.1 |
| seed | 42 |

学习率调度: warmup 线性上升 + cosine decay。

**启动命令**:
```bash
python -m src.training.train \
  --n_embd 512 --n_layer 10 --n_head 8 \
  --context_length 256 \
  --max_steps 20000 --batch_size 16 \
  --eval_interval 1000
```

### 4.2 微调

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| learning_rate | 1e-5 |
| min_lr | 1e-6 |
| max_steps | 10000 |
| warmup_steps | 20 |
| grad_clip | 1.0 |
| weight_decay | 0.1 |
| seed | 42 |

微调要点:
- 从 `pretrain_best.pt` 加载预训练权重
- 学习率比预训练小 (1e-5 vs 3e-4), 避免破坏已学知识
- 全参数微调, 不冻结任何层

**启动命令**:
```bash
python -m src.training.finetune \
  --max_steps 10000 --batch_size 16 \
  --eval_interval 100
```

## 5. 实验结果

### 5.1 预训练

| 指标 | 值 |
|------|-----|
| best val_loss | 1.2861 |
| test_loss | 1.4353 |

### 5.2 微调

| 指标 | 值 |
|------|-----|
| best val_loss | 1.2094 |
| test_loss | 2.0621 |

## 6. 评估与生成

### 6.1 解码参数

| 参数 | 值 |
|------|-----|
| temperature | 0.5 |
| top_k | 20 |
| max_new_tokens | 100 |

### 6.2 评估命令

```bash
python -m src.training.evaluate \
  --max_new_tokens 100 \
  --prompts "To be|Shall I|When in|Romeo|O my love" \
  --temperature 0.5 --top_k 20
```

## 7. Checkpoint 格式

符合分工文件 7.3 节要求, 每个 checkpoint 包含:

```python
{
    "model_state_dict": ...,          # 模型权重
    "optimizer_state_dict": ...,      # 优化器状态
    "step": ...,                      # 训练步数
    "best_val_loss": ...,             # 最佳验证 loss
    "config": {                       # 模型配置
        "vocab_size": ...,
        "context_length": ...,
        "n_embd": ...,
        "n_layer": ...,
        "n_head": ...,
        "dropout": ...,
    },
    "stage": "pretrain" / "finetune", # 训练阶段
    "tokenizer_hash": ...,            # 词表哈希
    "data_version": ...,              # 数据版本
    "parent_checkpoint": ...,         # 父 checkpoint (微调时指向预训练)
    "rng_states": ...,                # 随机数状态
}
```

## 8. 产物清单

| 产物 | 路径 |
|------|------|
| 预训练最佳模型 | `checkpoints/pretrain_best.pt` |
| 预训练最终模型 | `checkpoints/pretrain_last.pt` |
| 微调最佳模型 | `checkpoints/finetune_best.pt` |
| 微调最终模型 | `checkpoints/finetune_last.pt` |
| 词表 | `checkpoints/tokenizer.json` |
| 预训练日志 | `outputs/training/training_log.txt` |
| 预训练 loss 曲线 | `outputs/training/loss_curve.png` |
| 微调日志 | `outputs/finetune/finetune_log.txt` |
| 微调前后对比 | `outputs/finetune/before_after.txt` |
