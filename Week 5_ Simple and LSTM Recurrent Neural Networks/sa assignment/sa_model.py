import torch
from torch import nn

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        """
        初始化双向循环神经网络模型

        参数:
        vocab_size: 词汇表大小，即词汇的总数
        embed_size: 词嵌入的维度
        num_hiddens: 隐藏层的单元数
        num_layers: LSTM的层数
        """
        super(BiRNN, self).__init__(**kwargs) 
        self.embedding = nn.Embedding(vocab_size, embed_size) # 创建嵌入层，将词汇索引映射到词向量
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                           bidirectional=True)# 创建双向LSTM层
        self.decoder = nn.Linear(4 * num_hiddens, 2) # 创建线性层，将LSTM的输出映射到两个类别（正面和负面）

    def forward(self, inputs):
        """
        前向传播

        参数:
        inputs: 输入的词汇索引序列，形状为 (batch_size, seq_len)

        返回:
        outs: 模型的输出，即每个样本的预测类别得分，形状为 (batch_size, 2)
        """
        embeddings = self.embedding(inputs.T) # 对输入进行词嵌入，得到嵌入后的序列，形状为 (seq_len, batch_size, embed_size)
        self.rnn.flatten_parameters() # 确保LSTM的参数在使用DataParallel时正确初始化
        outputs, _ = self.rnn(embeddings)# 通过LSTM层，得到输出序列和隐藏状态，outputs形状为 (seq_len, batch_size, 2 * num_hiddens)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)# 将LSTM层的第一个时间步和最后一个时间步的输出拼接起来，形状为 (batch_size, 4 * num_hiddens)
        outs = self.decoder(encoding) # 通过线性层，将编码后的特征映射到两个类别，形状为 (batch_size, 2)
        return outs