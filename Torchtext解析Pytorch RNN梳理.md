# Torchtext解析/Pytorch RNN梳理

Torchtext组成部分：

![img](https://pic4.zhimg.com/80/v2-1e9cd89301ab3605481bb1720c536c2b_720w.jpg)来源Pytorch官网

其中较重要的三个部分：

- Field: 文本处理的通用参数设置，较重要的参数如下：

- - Field.sequential– Whether the datatype represents sequential data. If False, no tokenization is applied. Default: True.
  - use_vocab– Whether to use a Vocab object. If False, the data in this field should already be numerical. Default: True.
  - fix_length– A fixed length that all examples using this field will be padded to, or None for flexible sequence lengths. Default: None.
  - lower– Whether to lowercase the text in this field. Default: False.
  - tokenize– The function used to tokenize strings using this field into sequential examples. If “spacy”, the SpaCy tokenizer is used. If a non-serializable function is passed as an argument, the field will not be able to be serialized. Default: string.split.

- Dataset: 官网中有大量内置的Dataset去处理各种数据格式，其中TabularDataset可以很方便的读取CSV, TSV和JSON格式的文件。

- Iterator：torchtext到模型的输出，提供了我们对数据的一般处理方式。

Torthtext预处理流程：

- 定义Field
- 定义Dataset
- 建立vocab
- 构造迭代器，送入模型

## 1 准备数据

以kaggle上电影评论的情感分析为例，数据如下：

[https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/datawww.kaggle.com](https://link.zhihu.com/?target=https%3A//www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data)

train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.

The sentiment labels are:

0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

### **1.1 读取查看文件**

```python3
import pandas as pd
data = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
data.head()
test.head()
```

![img](https://pic4.zhimg.com/80/v2-cfd36b924af25c68b90c6e47bc2d3b4b_720w.jpg)

![img](https://pic3.zhimg.com/80/v2-3423f39c7f3ea69cca20f3655953da92_720w.jpg)

### 1.2 划分训练集和测试集

```text
from sklearn.model_selection import train_test_split
train, val = train_test_split(data, test_size=0.2)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
```

## 2 定义Field

导入需要的包和定义Pytorch张量使用的DEVICE

```text
import spacy
import torch
from torchtext import data, datasets
from torch import nn 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

首先定义Field参数中的tokenize，即分词函数，本文采用spacy库实现：

```text
spacy_en = spacy.load('en')

# create a tokenizer function
def tokenizer(text): 
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

> 在使用spacy时遇到错误：OSError: [E050] Can't find model 'en'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.[解决方案](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_39441762/article/details/86169771)

针对中文文本也可采取较为流行的jieba分词构造分词函数，例如：

```text
# create a tokenizer function
def tokenizer(text): 
    # regex为提前编译好的清洗文本用的正则表达式
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]
```

接着构造Field对象，Field在默认的情况下都期望一个输入是一组单词的序列，并且将单词映射成整数，这个映射被称为vocab。

如果一个field已经被数字化了并且不需要被序列化，可以将参数设置为use_vocab=False以及sequential=False，例如本文中的LABEL。

```text
LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
```

## 3 定义Dataset

对于csv/tsv类型的文件，TabularDataset很容易进行处理，故我们选它来生成Dataset。

由于训练集和验证机要做的预处理是一样的，因此可以使用split类方法来实现，其中的参数都较好理解，较重要的是参数是**fields：**

- Dataset.fields (dict[str, Field]) – Contains the name of each column or field, together with the corresponding Field object. Two fields with the same Field object will have a shared vocabulary.

对于像PhraseId这种在模型训练中不需要的特征，在构建Dataset的过程中可以直接使用None来代替。此外，fields中传入的(name, Field)必须与列的顺序相同。

```text
train,val = data.TabularDataset.splits(
        path='./', train='train.csv',validation='val.csv', format='csv',skip_header=True,
        fields=[('PhraseId',None),('SentenceId',None),('Phrase', TEXT), ('Sentiment', LABEL)])
test = data.TabularDataset('./test.tsv', format='tsv',skip_header=True,
        fields=[('PhraseId',None),('SentenceId',None),('Phrase', TEXT)])
```

查看生成的dataset：

```text
print(train[0])
print(train[0].__dict__.keys())
print(train[0].Phrase)
print(train[0].Sentiment)
```

可以看到train[0]输出为Example对象，其绑定了样本中的所有属性，其中Phrase部分已经被分词，但还没有转化为数字：

![img](https://pic3.zhimg.com/80/v2-1b74c2d337763e073a95d8b871cf0782_720w.png)



## 4 建立vocab

使用Field中的类方法build_vocab可以遍历训练集中绑定TEXT Field的数据，将单词注册到vocabulary中：

```text
# 构建vocabulary
TEXT.build_vocab(train)

print(TEXT.vocab)
print(type(TEXT.vocab.itos))
print(TEXT.vocab.itos[:10])
print(type(TEXT.vocab.stoi))
print(TEXT.vocab.stoi['the'])
```

可以看到已经定义了vocabulary object，其中vocabulary object有两个属性：

- Vocab.itos– A list of token strings indexed by their numerical identifiers.

- - itos即idx_to_word，是一个list，存放遍历得到的所有词汇

- Vocab.stoi– A collections.defaultdict instance mapping token strings to numerical identifiers.

- - stoi即word_to_idx，是一个字典，key为word，idx为value

![img](https://pic3.zhimg.com/80/v2-baa699f85ceea6fb30fbc841605b076e_720w.jpg)

通过len方法可以得到vocabulary的长度为15419，在后续构建Embedding层时会用到该数字。

```text
len_vocab = len(TEXT.vocab) 
len_vocab
```

截至目前，我们已经将每一个样本进行了分词，且将词转化为数字（vocabulary中的索引）。

------

除此之外，在建立词典的时候可以使用预训练的词向量，并自动构建embedding矩阵，如下所示，本文于此不过多赘述：

```text
TEXT.build_vocab(train, vectors='glove.6B.100d')
```

在使用预训练好的词向量时，我们需要在神经网络模型的Embedding层中明确地传递嵌入矩阵的初始权重。权重包含在词汇表的vectors属性中：

```text
# 通过Pytorch创建的Embedding层
embedding = nn.Embedding(2000, 256)
# 指定嵌入矩阵的初始权重
weight_matrix = TEXT.vocab.vectors
embedding.weight.data.copy_(weight_matrix )
```

## 5 构造迭代器

日常使用Pytorch训练网络时，每次训练都是输入一个batch，因此需要将前面得到的dataset转为迭代器，然后遍历迭代器获取batch输入。

torchtext中迭代器主要分为Iterator, BucketIerator, BPTTIterator三种。其中BucketIerator相比于标准迭代器，会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。但是在测试集中一般不想改变样本顺序，因此测试集使用Iterator迭代器来构建。

```text
train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase), 
                                 sort_within_batch=False, device=DEVICE)

val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.Phrase), 
                                 sort_within_batch=False, device=DEVICE)

test_iter = data.Iterator(dataset=test, batch_size=128, train=False,
                          sort=False, device=DEVICE)
```

其中sort_within_batch参数设置为True时，按照sort_key按降序对每个小批次内的数据进行排序。

> 如果我们需要padded序列使用pack_padded_sequence转换为PackedSequence对象时，这是非常重要的，我们知道如果想pack_padded_sequence方法必须将批样本按照降序排列。由此可见，torchtext不仅可以对文本数据进行很方变的处理，还可以很方便的和torchtext的很多内建方法进行结合使用。

其中sort_key: the BucketIterator needs to be told what function it should use to group the data.

**迭代器的使用**

准确的说刚刚构造的应该是可迭代对象，调用iter方法后才成为真正的迭代器对象。

```text
batch=next(iter(train_iter))
print(batch)
data = batch.Phrase
label = batch.Sentiment
print(data.shape)
print(data)
```

![img](https://pic4.zhimg.com/80/v2-6b89b81e243e4a70baf992bacbbb981b_720w.jpg)

```text
print(label.shape)
print(label)
```

![img](https://pic2.zhimg.com/80/v2-d2b6fff2e38ae2346afb9868d5a9b225_720w.jpg)

可以看到通过迭代器对象一次可以得到128个样本，其中Phrase的维度为(seq_len,batch_size)，即每列代表一个样本（一个句子，这个批次中每句为33个词），Sentiment的维度为seq_len，即每个样本的Sentiment为一个标量。

## 6 构建模型

```text
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)
        self.rnn = nn.RNN(100,128)
        self.linear = nn.Linear(128,5)

    def forward(self, x):
        seq_len , batch_size = x.shape # x.shape=(seq_len, batch_size)
        vec = self.embedding(x) # vec的维度为(seq_len, batch_size, 100)
        output,hidden = self.rnn(vec) # RNN初始化的hidden如果不提供则默认为全0张量 
        # output的维度 (seq_len, batch, hidden_size 128)
        # hidden的维度 (numlayers 1, batch_size, hidden_size 128)
        out = self.linear(hidden.view(batch_size, -1))
        return out
        # out的维度 (batch_size, 5)
```

### 6.1 Embedding层维度

![img](https://pic1.zhimg.com/80/v2-5751be98f916676894ea147ae6e16668_720w.jpg)

可以看到对于Embedding层来说，不管输入维度是多少，都会在最后增加一个Embedding的维度，本例中，通过Embedding层后，维度为(seq_len, batch_size 128, 100)

### 6.2 RNN层维度

Pytorch中有两种方式实现RNN，分别是torch.nn中的RNNCell和RNN，本文采用nn.RNN。

借用[Liu Hongpu](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Y7411d7Ys)老师的两张图，可以十分清晰的阐述RNN中的维度部分：

![img](https://pic2.zhimg.com/80/v2-1c62ae4a6833b2f98d5220feca7a653d_720w.jpg)

![img](https://pic3.zhimg.com/80/v2-c3806cc8b4124441219e75c1579292f6_720w.jpg)

由于本文是一个分类问题，因此针对RNN的输出，选用最后一个时间点输出的hidden即可。

### 6.3 线性层维度

![img](https://pic3.zhimg.com/80/v2-7c6a7d79d30f860abb8042062146b462_720w.jpg)

可见，nn.Linear需要输入的第一个维度为batch_size，最后一个维度为样本的维度，因此需要对RNN的输出做一个变形：hidden.view(batch_size, -1)。其输出的维度为(batch_size, 5)。

### 6.4 模型的训练

分类问题选择交叉熵作为损失函数，并采取Adam进行优化：

```text
criterition = torch.nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.05)
```

> torch.nn.CrossEntropyLoss() combines nn.LogSoftmax() and nn.NLLLoss().

模型的训练部分较常规且不是本文重点，因此只写了训练数据相关，代码如下：

```text
n_epoch = 20

for epoch in range(n_epoch):
    for bacth_idx, batch in enumerate(train_iter):
        data = batch.Phrase.to(DEVICE)
        target = batch.Sentiment.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterition(outputs, target)
        loss.backward()
        optimizer.step()
        
        if (bacth_idx+1)%200 == 0:
            _,y_pred = torch.max(outputs, -1)
            acc = torch.mean((torch.tensor(y_pred == target,dtype=torch.float)))
            print('epoch: %d \t batch_id : %d \t loss: %.4f \t train_acc: %.4f'
                  %(epoch, bacth_idx+1, loss, acc))   
```