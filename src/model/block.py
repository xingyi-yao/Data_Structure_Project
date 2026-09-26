import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module): #定义一个神经网络模块
    """一个 Transformer Block,入输出均为 [B, T, n_embd]。""" #但是里面数据已经经过Transformer Block的处理，已经混合了其他token的信息

    def __init__(self, n_embd, context_length, dropout=0.1, n_head=1):
        super().__init__()

        # Attention 前的归一化
        self.ln1 = nn.LayerNorm(n_embd) #每个token自己的64个数字进行归一化，
        self.attention = CausalSelfAttention(
            n_embd=n_embd,
            context_length=context_length,
            dropout=dropout,
            n_head=n_head,
        )
        self.attn_output_dropout = nn.Dropout(dropout)

        # FFN 前的归一化
        self.ln2 = nn.LayerNorm(n_embd)
        self.feedforward = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):
        # 第一部分：归一化 → Attention → 残差相加
        attention_output = self.attention(self.ln1(x)) #把x规范化之后送入Attention
        x = x + self.attn_output_dropout(attention_output)  #在原来基础上补充一些信息

        # 第二部分：归一化 → FFN → 残差相加
        x = x + self.feedforward(self.ln2(x))

        return x


# if __name__ == "__main__":
#     torch.manual_seed(42)

#     block = TransformerBlock(
#         n_embd=64,
#         context_length=16,
#         dropout=0.0,
#         n_head=4,
#     )

#     # requires_grad=True，让我们也能检查输入的梯度。
#     x = torch.randn(2, 8, 64, requires_grad=True)
#     y = block(x)

#     print("输入形状：", x.shape)
#     print("输出形状：", y.shape)

#     assert y.shape == x.shape
#     assert torch.isfinite(y).all()
#     print("Block 前向检查通过！")

#     # 仅用于检查梯度能否传播，不是正式的语言模型训练损失。
#     test_loss = y.square().mean()
#     test_loss.backward()

#     assert x.grad is not None
#     assert torch.isfinite(x.grad).all()

#     for name, parameter in block.named_parameters():
#         assert parameter.grad is not None, f"{name} 没有梯度"
#         assert torch.isfinite(parameter.grad).all(), f"{name} 梯度异常"

#     print("Block 反向传播检查通过！")