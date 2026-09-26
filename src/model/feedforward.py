import torch #导入 PyTorch 库
import torch.nn as nn #导入Pytorch的神经网络工具并给他一个简称，之后的Linear,Dropout,GeLU都是直接写好的神经网络组件

class FeedForward(nn.Module): #这是一个前馈神经网络的类，继承了nn.Module类
    """对每个 token 的特征向量进行变换，保持输入输出形状一致。"""

    def __init__(self, n_embd, dropout=0.1): # n_embd是一个token用几个数字表示
        super().__init__() #初始化父类nn.Module的基础功能，就是调用父类的构造函数，初始化PyTorch的神经网络基础功能
        self.net = nn.Sequential( #这里是按顺序执行的意思，数据会先经过Linear,再经过GELU，再经过Linear，再经过Dropout
            nn.Linear(n_embd, 4 * n_embd),  # 把信息展开到更大空间中进行处理，每一个新的数字都是原来64个加权计算出来的，权重由训练学习决定
            nn.GELU(), #当某些特征出现时，就需要强化某些信息，弱化某些信息 非线性变换，如果是一个比较大的正数（特征重要），基本就会保留下来；如果是小正数（特征不明显），会保留一部分；如果是负数，大部分会被压制（这个特征是负向的，所以要大幅抑制）
            nn.Linear(4 * n_embd, n_embd),  # 因为后面需要和残差连接起来，所以需要恢复原来维度，具体怎么压缩也是训练出来的
            nn.Dropout(dropout), # 会在64个维度里面随机挑选几个维度的数字置0，就是随机的，不是说把小的数字丢弃，这样可以避免过拟合；并且把留下的值适当放大，保持整体期望大小不变。
        )

    def forward(self, x):
        return self.net(x) #Sequential 负责定义里面这些层怎么依次连接，而 forward() 负责定义整个模块收到输入 x 以后，整体怎么计算输出


# 直接运行这个文件时，执行下面的简单检查。
# 其他文件导入 FeedForward 时，不会执行这里。
# if __name__ == "__main__":
#     ffn = FeedForward(n_embd=64, dropout=0.0)

#     # 模拟：2 条序列，每条 8 个 token，每个 token 有 64 个特征。
#     x = torch.randn(2, 8, 64)
#     y = ffn(x)

#     print("输入形状：", x.shape)
#     print("输出形状：", y.shape)

#     assert y.shape == x.shape
#     assert torch.isfinite(y).all()
#     print("FFN 前向检查通过！")