import torch
import torch.nn as nn
import torch.nn.functional as F
from src.inference.kv_cache import append_layer_kv


class CausalSelfAttention(nn.Module):
    """支持单头或多头的因果自注意力。"""

    def __init__(self, n_embd, context_length, dropout=0.1, n_head=1): #n_embd就是之前的一个token需要几个数字表示，context_length是一次性最多处理几个token,n_head是几个注意力头，平均分维度数，不同的头学习不同类型的信息，比如语法关系，位置关系之类的，但这不是分配到的，是训练过程中模型自己决定的
        super().__init__()

        if n_head <= 0 or n_embd % n_head != 0:
            raise ValueError("n_head 必须为正整数，且 n_embd 能被 n_head 整除")

        self.context_length = context_length #这里保存文本最大长度
        self.n_head = n_head #这里保存注意力头数
        self.head_dim = n_embd // n_head  #算出来一个注意力头负责多少维度
        self.scale = self.head_dim ** -0.5 #

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x, past_kv=None, use_cache=False):
        if past_kv is not None and not use_cache:
            raise ValueError("past_kv 需要 use_cache=True")
        B, T, D = x.shape
        if not 1 <= T <= self.context_length:
            raise ValueError("序列长度必须在 1 和 context_length 之间")

        # [B,T,D] → [B,T,H,head_dim] → [B,H,T,head_dim]
        q = self.query(x).reshape(B, T, self.n_head, self.head_dim) #把总共的维度分给每个注意力头上
        k = self.key(x).reshape(B, T, self.n_head, self.head_dim)
        v = self.value(x).reshape(B, T, self.n_head, self.head_dim)
        q = q.transpose(1, 2) #交换这两个维度
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k, v = append_layer_kv(past_kv, k, v)
        past_length = k.size(-2) - T
        if k.size(-2) > self.context_length:
            raise ValueError("缓存与输入总长度超过 context_length")

        # 各个头分别计算注意力，分数形状为 [B,H,T,T]。
        scores = (q @ k.transpose(-2, -1)) * self.scale #也是交换两个维度，但是使用的负标签
        scores = scores.masked_fill(
            ~self.causal_mask[past_length:past_length + T, :k.size(-2)], float("-inf")
        )           #只保留三角的True部分，其他部分为False，False部分的分数被置为负无穷大，这样在softmax后就会变成0
        weights = torch.softmax(scores, dim=-1)# 把注意力分数转化为注意力权重
        weights = self.attn_dropout(weights)
        output = weights @ v #把其他维的信息混合进来

        # [B,H,T,head_dim] → [B,T,H,head_dim] → [B,T,D]
        output = output.transpose(1, 2).contiguous().view(B, T, D) #变回原来的维度数
        output = self.proj(output)
        return (output, (k, v)) if use_cache else output


class SDPAAttention(CausalSelfAttention):
    """性能实验变体；保留独立 Q/K/V 参数名以直接复用基线权重。"""

    def forward(self, x):
        B, T, D = x.shape
        if not 1 <= T <= self.context_length:
            raise ValueError("序列长度必须在 1 和 context_length 之间")
        q, k, v = [layer(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
                   for layer in (self.query, self.key, self.value)]
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0)
        return self.proj(output.transpose(1, 2).contiguous().view(B, T, D))


# if __name__ == "__main__":
#     torch.manual_seed(42)
#     attention = CausalSelfAttention(
#         n_embd=64, context_length=16, dropout=0.0, n_head=4
#     )
#     attention.eval()
#     x = torch.randn(2, 8, 64)

#     with torch.no_grad():
#         y = attention(x)
#         assert y.shape == x.shape
#         assert torch.isfinite(y).all()
#         print("Output shape:", y.shape)
#         print("Heads:", attention.n_head)
#         print("Head dimension:", attention.head_dim)

#         x_changed = x.clone()
#         x_changed[:, 4:, :] = torch.randn_like(x_changed[:, 4:, :])
#         y_changed = attention(x_changed)
#         torch.testing.assert_close(
#             y[:, :4, :], y_changed[:, :4, :], rtol=1e-5, atol=1e-6
#         )
#         print("Multi-head attention and causal mask: OK")
