# 数据说明

## 目录结构

```
data/
├── raw/                # 原始文本数据
│   ├── pretrain.txt    # 预训练数据（莎士比亚戏剧全集）
│   ├── finetune.txt    # 微调数据（莎士比亚十四行诗 154 首）
│   └── finetune.txt.bak# 微调数据原始备份（未重整格式）
└── README.md           # 本文件
```

## 数据来源

### 预训练数据：pretrain.txt
- **来源**：Project Gutenberg 的莎士比亚全集（戏剧部分）
- **内容**：Coriolanus, Hamlet, Macbeth, Romeo and Juliet 等戏剧
- **字符数**：约 1,115,394 字符
- **用途**：学习基础英语语言规律（词法、句法、常见表达）

### 微调数据：finetune.txt
- **来源**：Project Gutenberg 的莎士比亚十四行诗（The Sonnets）
- **内容**：154 首十四行诗（I ~ CLIV）
- **字符数**：约 96,024 字符
- **用途**：学习诗歌风格（韵律、修辞、意象）

## 清洗规则

### pretrain.txt
1. 从 Gutenberg 完整文本中提取戏剧部分
2. 保留原始换行、对白格式
3. 未做额外清洗（戏剧文本本身较干净）

### finetune.txt
1. 从 Gutenberg 文本中按罗马数字标题（I, II, ..., CLIV）切分出 154 首
2. 删除文件末尾的 Project Gutenberg 许可声明
3. 删除 `***` 分隔符
4. **重整格式**：把每首诗的"标题 + 空行 + 内容"合并为一个段落块，
   诗与诗之间用空行 `

` 分隔
   （这样 `dataset.py` 的 `split_by_line=True` 可以按整首诗划分）

## 去重记录

### 交叉去重检查（分工 5.2 节第 3 条）
- **检查方法**：对每首十四行诗取前 100 字符作为指纹，
  在预训练文本中搜索是否出现
- **结果**：154 首十四行诗中 **0 首** 出现在预训练数据中
- **结论**：预训练数据（戏剧）与微调数据（十四行诗）**无重叠**，
  无数据泄漏

### 著名诗句验证
检查了以下著名十四行诗开头，均**未**出现在 pretrain.txt 中：
- "Shall I compare thee to a summer" (Sonnet 18)
- "Let me not to the marriage of true minds" (Sonnet 116)
- "When in disgrace with fortune and men" (Sonnet 29)
- "From fairest creatures we desire increase" (Sonnet 1)

## 数据划分

### 划分比例（分工 5.2 节第 1 条）
按 **90% / 5% / 5%** 划分 train / val / test

### 划分方式
- **预训练数据**：按 token 位置连续切分（`split_by_line=False`）
  适合连续文本，不打乱文本顺序
- **微调数据**：按整首诗划分（`split_by_line=True`）
  154 鮨鹉诗为整体单元，不会把一首诗切散到不同 split

### 微调数据划分结果
- 训练集：138 首诗（86,129 tokens，89.7%）
- 验证集：7 首诗（4,289 tokens，4.5%）
- 测试集：9 首诗（5,595 tokens，5.8%）

## 词表

- **建立方式**：用预训练 + 微调数据的字符合并建立（分工要求）
- **词表大小**：88
- **UNK/EOS**：当前 CharTokenizer 无 UNK/EOS 接口，
  通过合并建词表避免词表外字符报错（分工 5.2 节第 6 条的 EOS 待补）
