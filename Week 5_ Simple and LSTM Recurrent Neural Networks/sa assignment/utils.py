import torch
import collections
import jieba
from torch.utils import data

class Vocab:
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None) -> None:
        """
        初始化词汇表对象

        参数:
        tokens: 用于构建词汇表的令牌列表
        min_freq: 词汇表中包含的最低频率
        reserved_tokens: 预留的特殊令牌
        """
        if tokens == None:
            tokens =[]
        if reserved_tokens == None:
            reserved_tokens = []
        
         # 统计令牌频率
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 初始化索引到令牌的映射和令牌到索引的映射
        self.idx_to_token =  reserved_tokens + ['<unk>']
        # print(self.idx_to_token)
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}
       
        # 添加满足频率要求的令牌到词汇表
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)# 返回词汇表大小
    
    def __getitem__(self, tokens):
        """
        根据令牌或令牌列表，返回相应的索引或索引列表

        参数:
        tokens: 令牌或令牌列表

        返回:
        相应的索引或索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        """
        根据索引或索引列表，返回相应的令牌或令牌列表

        参数:
        indices: 索引或索引列表

        返回:
        相应的令牌或令牌列表
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property        
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs # 返回令牌频率 
    
def count_corpus(tokens): 
    """
    统计令牌频率

    参数:
    tokens: 令牌列表或令牌列表的列表

    返回:
    令牌频率计数器
    """
    # print(tokens)
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充序列

    参数:
    line: 序列
    num_steps: 目标长度
    padding_token: 填充令牌

    返回:
    截断或填充后的序列
    """
    if len(line) > num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps-len(line))

def load_array(data_arrays, batch_size, is_train=True):
    """
    构建PyTorch数据迭代器

    参数:
    data_arrays: 数据数组
    batch_size: 批次大小
    is_train: 是否为训练模式

    返回:
    数据加载器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def splitTrain_test(dataset, training_rate=0.9):
    """
    将数据集分为训练集和测试集

    参数:
    dataset: 数据集
    training_rate: 训练集比例

    返回:
    训练集和测试集
    """
    num_samples = len(dataset)
    num_training = int(num_samples * training_rate)
    # num_testing = num_samples - num_training
    
    return dataset[:num_training], dataset[num_training:]

def load_SA_data(pos_dir, neg_dir, batch_size, max_len):
    """
    加载情感分析数据

    参数:
    pos_dir: 正面情感数据文件路径
    neg_dir: 负面情感数据文件路径
    batch_size: 批次大小
    max_len: 序列最大长度

    返回:
    训练数据迭代器，测试数据迭代器，词汇表
    """
    # 读取正面和负面情感数据
    pos_data = [(line.strip(), 1) for line in open(pos_dir, 'rt', encoding='UTF-8')]
    neg_data = [(line.strip(), 0) for line in open(neg_dir, 'rt', encoding='UTF-8')]
    
    # 合并数据和标签
    dataset_labels = pos_data + neg_data
    dataset = [sent[0] for sent in dataset_labels]
    labels = [sent[1] for sent in dataset_labels]
    
    # 对数据进行分词
    dataset = tokenize_sentences(dataset)
    vocab = Vocab(dataset, min_freq=1, reserved_tokens=['<pad>', '<sos>', '<eos>'])
    
    # 划分训练集和测试集
    train_data, test_data = splitTrain_test(dataset)
    train_label, test_label = splitTrain_test(labels)
    
    # 将数据转换为索引并进行截断或填充
    trainData_idxs = torch.tensor([truncate_pad(vocab[sent], max_len, vocab['<pad>']) for sent in train_data])
    testData_idxs = torch.tensor([truncate_pad(vocab[sent], max_len, vocab['<pad>']) for sent in test_data])
    
    # 构建数据迭代器
    train_iter = load_array((trainData_idxs, torch.tensor(train_label)), batch_size)
    test_iter = load_array((testData_idxs, torch.tensor(test_label)), batch_size, is_train=False)
    
    return train_iter, test_iter, vocab
    

def tokenize_sentences(input_data):
    """
    对输入数据进行分词

    参数:
    input_data: 输入数据

    返回:
    分词后的数据
    """
    tokenized_sents = []
    for sent in input_data:
        tokenized_sents.append(tokenize(sent))
    
    return tokenized_sents

def tokenize(sentence):
    """
    对句子进行分词

    参数:
    sentence: 输入句子

    返回:
    分词后的词列表
    """
    sentence = sentence.strip()
    words = jieba.lcut(sentence, cut_all=False)
    return words

if __name__ == '__main__':
    
     # 数据目录和文件名
    data_dir = 'C:\\Users\\Lenovo\\Desktop\\sa assignment\\'
    pos_file = 'pos.csv'
    neg_file = 'neg.csv'
    
     # 加载情感分析数据
    train_iter, test_iter, vocab = load_SA_data(data_dir+pos_file, data_dir+neg_file, 16, 100)
    # print(vocab)
    # print(train_iter)
    for i, (feature, label) in enumerate(train_iter):
        
        # if i == 10:
        #     break
        
        print(i, feature.shape, label.shape)
        break