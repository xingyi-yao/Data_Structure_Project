import torch
import torch.nn as nn
import torch.nn.functional as F #这个函数用来计算训练时候的交叉熵损失

from .block import TransformerBlock


class GPT(nn.Module): #定义整个GPT模型
    """最小 GPT:输入 token ID,输出 logits 和可选的训练损失。"""

    def __init__(self, vocab_size, context_length=128,
                 n_embd=64, n_layer=2, dropout=0.1, n_head=1): #vocab_size是统计词表里有多少token n_layer是有几个Transformer Block
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.config = dict(
            vocab_size=vocab_size, context_length=context_length,
            n_embd=n_embd, n_layer=n_layer, dropout=dropout, n_head=n_head,
        ) #创建了一个Python字典，保存模型的配置

        # token ID → 特征向量
        self.token_embedding = nn.Embedding(vocab_size, n_embd) #根据词表转换为后面可以处理的向量

        # 位置编号 → 位置向量
        self.position_embedding = nn.Embedding(context_length, n_embd)
        self.embedding_dropout = nn.Dropout(dropout) #要给每一个token进行位置编号，这些编号也被转化为向量

        # 创建多个独立的 Transformer Block
        self.blocks = nn.ModuleList([ #专门用来存n_layer个Transformer Block的列表
            TransformerBlock(n_embd, context_length, dropout, n_head=n_head)
            for _ in range(n_layer)
        ])

        self.final_norm = nn.LayerNorm(n_embd)

        # 特征向量 → 词表中每个 token 的预测分数
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError("idx 的形状必须是 [B, T]")

        _, T = idx.shape
        if not 1 <= T <= self.context_length:
            raise ValueError("序列长度必须在 1 和 context_length 之间")

        positions = torch.arange(T, device=idx.device) 

        x = self.token_embedding(idx) + self.position_embedding(positions) #就是把token的向量表示和它的位置向量表示加起来
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x)) #对下一个token预测分数

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets 和 idx 的形状必须相同")

            # targets 已经由数据层右移；这里不再移动。
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss


if __name__ == "__main__":
    torch.manual_seed(42)

    model = GPT(
        vocab_size=100,
        context_length=16,
        n_embd=64,
        n_layer=2,
        dropout=0.0,
        n_head=4,
    )

    # 9 个 token 错开一位，得到 8 个输入和对应标签。
    tokens = torch.randint(0, 100, (2, 9))
    idx = tokens[:, :-1]
    targets = tokens[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    # 保存更新前的参数，用于检查参数是否真的改变。
    before = model.lm_head.weight.detach().clone()

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, 100)
    assert torch.isfinite(logits).all()
    assert loss.ndim == 0 and torch.isfinite(loss)

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    # 反向传播，检查各参数的梯度。
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    # 更新一次参数。
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight.detach())
    print("Forward, backward and parameter update: OK")

    # 检查不提供标签时的调用方式。
    model.eval()
    with torch.no_grad():
        scores, no_loss = model(idx)

    assert scores.shape == (2, 8, 100)
    assert no_loss is None
    print("Inference interface: OK")