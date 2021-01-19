# Bert预训练细节

## 任务一：Masked LM

![img](C:\Users\Administrator\AppData\Local\YNote\data\qqA4E85D9283EA896834A3F05C6F20D048\bc6086f0923a4a8ea925ba3a30ac49b9\clipboard.png)

Bert预训练数据准备代码

```
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
#语料库
text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)
#构建词典

token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)
    
#以上code是从语料库建立字典的过程
#sentences=['hello how are you i am romeo', 'hello romeo my name is juliet nice to meet you', 'nice meet you too how are you today', 'great my baseball team won the competition', 'oh congratulations juliet', 'thanks you romeo']
#token_list=[[20, 24, 16, 11, 13, 19, 10], [20, 10, 15, 21, 4, 18, 7, 22, 27, 11], [7, 27, 11, 6, 24, 16, 11, 12], [28, 15, 14, 8, 25, 9, 23], [5, 26, 18], [17, 11, 10]]
#word_dict={'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, 'is': 4, 'oh': 5, 'too': 6, 'nice': 7, 'team': 8, 'the': 9, 'romeo': 10, 'you': 11, 'today': 12, 'i': 13, 'baseball': 14, 'my': 15, 'are': 16, 'thanks': 17, 'juliet': 18, 'am': 19, 'hello': 20, 'name': 21, 'to': 22, 'competition': 23, 'how': 24, 'won': 25, 'congratulations': 26, 'meet': 27, 'great': 28}
#number_dict={0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]', 4: 'is', 5: 'oh', 6: 'too', 7: 'nice', 8: 'team', 9: 'the', 10: 'romeo', 11: 'you', 12: 'today', 13: 'i', 14: 'baseball', 15: 'my', 16: 'are', 17: 'thanks', 18: 'juliet', 19: 'am', 20: 'hello', 21: 'name', 22: 'to', 23: 'competition', 24: 'how', 25: 'won', 26: 'congratulations', 27: 'meet', 28: 'great'}

    
    
    

maxlen = 30  # maximum of length  句子的最大输入长度
batch_size = 6
max_pred = 5  # 设置最大预测词的个数 max tokens of prediction
n_layers = 6  # number of Encoder of Encoder Layer
n_heads = 12  # number of heads in Multi-Head Attention
d_model = 768  # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
#sample IsNext and NotNext to be same in small batch size
#positive 代表B is A下一句，negative 代表B is not A下一句,positive+negative的总数目组成一个batch
def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)  #随机排序
        masked_tokens, masked_pos = [], []
        #已经随机排序了，相当于随机采样
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 随机数很大程度都会小于0.8
                input_ids[pos] = word_dict['[MASK]'] # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace
        #The model is trained with both Masked LM and Next Sentence Prediction together.
        # This is to minimize the combined loss function of the two strategies — “together is better”.
        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        #采样相邻的句子tokens_a_index + 1 == tokens_b_index
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2: #IsNext
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:#NotNext
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
#Proprecessing Finished
batch=make_batch()
print(batch)
```

## 任务二：Next Sentence Prediction

![img](C:\Users\Administrator\AppData\Local\YNote\data\qqA4E85D9283EA896834A3F05C6F20D048\f80a79ea829443c2839bbf4cf63fb78e\clipboard.png)