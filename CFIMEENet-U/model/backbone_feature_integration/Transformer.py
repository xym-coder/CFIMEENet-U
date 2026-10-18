import torch
from torch import nn, Tensor
import numpy as np
from config.config import config as cfg

# 位置编码模块，为序列数据添加位置信息，使Transformer能够感知元素的顺序
class Positional_encoding(nn.Module):
    def __init__(self, PE_size, n_position=256):
        super(Positional_encoding, self).__init__()
        self.PE_size = PE_size
        self.n_position = n_position
        self.register_buffer('pos_table', self.get_encoding_table(n_position, PE_size))
    # 生成位置编码表
    def get_encoding_table(self, n_position, PE_size):
        # 计算每个位置的编码值
        position_table = np.array(
            [[pos / np.power(10000, 2. * i / self.PE_size) for i in range(self.PE_size)] for pos in range(n_position)])
        # 偶数维度使用sin，奇数维度使用cos
        position_table[:, 0::2] = np.sin(position_table[:, 0::2])
        position_table[:, 1::2] = np.cos(position_table[:, 1::2])
        return torch.FloatTensor(position_table).unsqueeze(0)

    def forward(self, inputs):
        # 将位置编码直接加到输入特征上
        return inputs + self.pos_table[:, :inputs.size(1), :].clone().detach()

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1, if_resi=True, batch_first=False):
        super(MultiHeadAttention, self).__init__()
        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)
         # 调用 PyTorch 官方的 MultiheadAttention
        self.MultiheadAttention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        # 对 Query, Key, Value 分别做一个线性映射并加入 ReLU 激活
        self.Q_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.if_resi = if_resi

    def forward(self, inputs):
        query = self.layer_norm(inputs)
        # Q/K/V投影
        q = self.Q_proj(query)
        k = self.K_proj(query)
        v = self.V_proj(query)
        # 自注意力计算
        attn_output, attn_output_weights = self.MultiheadAttention(q, k, v)
        if self.if_resi:
            attn_output += inputs
        else:
            attn_output = attn_output

        return attn_output

# 前馈神经网络,两层全连接,中间带 ReLU
class FeedForward(nn.Module):
    def __init__(self, in_channel, FFN_channel, if_resi=True):
        super(FeedForward, self).__init__()
        """
        1024 2048
        """
        output_channel = (FFN_channel, in_channel)
        # 第一层升维
        self.fc1 = nn.Sequential(nn.Linear(in_channel, output_channel[0]), nn.ReLU())
        # 第二层降维回原始维度
        self.fc2 = nn.Linear(output_channel[0], output_channel[1])
        # 层归一化
        self.layer_norm = nn.LayerNorm(in_channel)
        self.if_resi = if_resi

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        if self.if_resi:
            outputs += inputs
        else:
            outputs = outputs
        return outputs

# Transformer编码器层(包含多个MHA+FFN块)
class TransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attention_size,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=True, block_nums=3, batch_first = False):
        super(TransformerLayer, self).__init__()
        self.block_nums = block_nums
        self.if_resi = if_resi
        # 调整输入维度到注意力处理维度
        self.linear = nn.Linear(in_dim, attention_size)
        # 动态创建多个Transformer块
        for i in range(self.block_nums):
            # 多头自注意力模块
            self.__setattr__('MHA_self_%d' % i, MultiHeadAttention(num_heads, attention_size, dropout=drop_rate, if_resi=if_resi, batch_first=batch_first))
            # 前馈网络模块
            self.__setattr__('FFN_%d' % i, FeedForward(out_dim, dim_feedforward, if_resi=if_resi))

    def forward(self, query):
        inputs = self.linear(query)
        # outputs = inputs
         # 依次通过每个Transformer块
        for i in range(self.block_nums):
            outputs = self.__getattr__('MHA_self_%d' % i)(inputs)  # 自注意力
            outputs = self.__getattr__('FFN_%d' % i)(outputs)  # 前馈网络
            if self.if_resi:
                inputs = inputs+outputs
            else:
                inputs = outputs
        # outputs = inputs
        return inputs

# 轮廓预测Transformer模块
class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=False, block_nums=3,
                 pred_num=2, batch_first=False):
        super().__init__()
        # 批归一化
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        # 卷积分支保留局部特征
        self.conv1 = nn.Conv1d(in_dim, out_dim, 1, dilation=1)
        # Transformer分支捕获全局依赖
        self.transformer = TransformerLayer(in_dim, out_dim, num_heads, attention_size=out_dim,
                                            dim_feedforward=dim_feedforward, drop_rate=drop_rate,
                                            if_resi=if_resi, block_nums=block_nums, batch_first=batch_first)
        # self.transformer_contour = TransformerLayer(in_dim, out_dim, num_heads, attention_size=out_dim,
        #                                     dim_feedforward=dim_feedforward, drop_rate=drop_rate,
        #                                     if_resi=if_resi, block_nums=block_nums, batch_first=True)
        # 预测头,多层1x1卷积组合的MLP
        self.prediction = nn.Sequential(
            nn.Conv1d(2*out_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, pred_num, 1))

    # 前向传播
    def forward(self, x):
        x = self.bn0(x)

        # ver1
        x1 = x.permute(0, 2, 1)
        x1 = self.transformer(x1)
        x1 = x1.permute(0, 2, 1)
        x = torch.cat([x1, self.conv1(x)], dim=1)
        pred = self.prediction(x)

        # ver2
        # x = x.permute(0, 2, 1)
        # x1 = self.transformer(x)
        # x2 = self.transformer_contour(x)
        # x1 = x1.permute(0, 2, 1)
        # x2 = x2.permute(0, 2, 1)
        # x = torch.cat([x1, x2], dim=1)
        # pred = self.prediction(x)

        return pred



