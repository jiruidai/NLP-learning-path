import torch
from torch import nn
from d2l import torch as d2l
from sa_model import BiRNN
from utils import load_SA_data, tokenize

#定义超参数
batch_size = 64 # 每个批次的样本数
max_len = 150 # 序列的最大长度
data_dir = './data/' # 数据目录
pos_file = 'pos.csv' # 正面情感
neg_file = 'neg.csv' # 负面情感
lr, num_epoch = 0.001, 5 # 学习率和训练的轮数

# 定义模型参数
embed_size, num_hiddens, num_layers = 300, 256, 3  # 嵌入层大小、隐藏单元数、LSTM层数
devices = d2l.try_all_gpus() # 尝试获取所有可用的GPU

# 如果没有可用的 GPU，使用 CPU，本任务使用的是CPU
if not devices:
    devices = [torch.device('cpu')]

# 初始化模型权重
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

# 执行一个批次的训练
def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X] # 将数据移到第一个设备
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train() # 将模型设置为训练模式
    trainer.zero_grad() # 清空梯度
    pred = net(X) # 前向传播
    l = loss(pred, y) # 计算损失
    l.sum().backward() # 反向传播
    trainer.step() # 更新参数
    train_loss_sum = l.sum() # 累计训练损失
    train_acc_sum = d2l.accuracy(pred, y) # 累计训练准确率
    return train_loss_sum, train_acc_sum

# 训练模型
def train_epochs(net, train_iter, test_iter, loss, trainer, num_epochs, devices):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]) # 使用DataParallel进行多GPU训练
    for epoch in range(num_epochs):
        # 训练过程中的累计指标（损失、准确率、样本数、预测数）
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print('batch: {0}, loss: {1}, accuracy: {2}'.format(i + 1, metric[0] / metric[2], metric[1] / metric[3]))
                # animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter) # 在测试集上评估
        print('epoch: {0}, test accuracy: {1}'.format(epoch + 1, test_acc))
        # animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {metric[0] / metric[2]:.3f}, train acc ' f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' f'{str(devices)}')

# 预测文本序列的情感
def predict_sentiment(net, sequence):
    """Predict the sentiment of a text sequence."""
    # sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

# 主函数
if __name__ == '__main__':
    # 加载数据
    train_iter, test_iter, vocab = load_SA_data(data_dir + pos_file, data_dir + neg_file, batch_size, max_len)
    # 初始化模型
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    net.apply(init_weights)

    # 训练模型
    train_epochs(net, train_iter, test_iter, loss, trainer, num_epoch, devices)

    # 在测试数据上进行预测
    for i, (feat, label) in enumerate(test_iter):
        for i in range(len(feat)):
            one_sample = feat[i, :]
            # print(one_sample.tolist())
            print('sent: ', ''.join(vocab.to_tokens(one_sample.tolist())), 'label: ',
                  'pos' if label[i].tolist() == 1 else 'neg', 'predict: ', predict_sentiment(net, one_sample))
        break