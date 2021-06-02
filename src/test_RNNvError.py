#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import random
import time
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor.dataset.dataset import Dataset


# In[2]:


def decode_file(file_ID_name, file_result_name):
    data = []
    line1 = "this is just a test sentence"
    line2 = "1,0,0,0,0,0,0,1"
    feature = line1.replace('\n','').lower().split(' ')
    label = line2.replace('\n','').lower().split(',')
    label = [int(x) for x in label]
    data.append([feature, label[1:]])
    
    random.shuffle(data)
    return data


# In[3]:


folder_name = "/.cached/data/"
train_ID_name = folder_name + "ID_train"
train_result_name = folder_name + "ISEAR_train"
train_data = decode_file(train_ID_name, train_result_name)


# In[4]:


class MyVocab:
    def __init__(self):
        self._vocab = {}
        self._size = 1
        
    def insert(self, word):
        if word not in self._vocab.keys():
            self._vocab[word] = self._size
            self._size = self._size + 1
    
    def locate(self, word):
        if word not in self._vocab.keys():
            return 0
        return self._vocab[word]
    
    def size(self):
        return self._size

def build_vocab(data):
    vocab = MyVocab()
    for sentence in data:
        for word in sentence[0]:
            vocab.insert(word)
    return vocab

def resize_sentence(data, normal_len):
    def pad(sentence, size):
        return sentence[:size] if len(sentence) > size else sentence+[0]*(size-len(sentence))
    return [[pad(sentence[0], normal_len), sentence[1]] for sentence in data]
    

train_vocab = build_vocab(train_data)


# In[5]:


class TrainDataset(Dataset):
    def __init__(self, vocab=None, data=None, normal_len=180, batch_size=1, shuffle=False):
        super().__init__()
        
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._total_len = len(data)
        self._sentence_len = normal_len
        self._vocab = vocab
        self._data = []
        for sentence in data:
            self._data.append([[vocab.locate(x) for x in sentence[0]], sentence[1]])
        self._data = resize_sentence(self._data, normal_len)
        self.set_attrs(batch_size=self._batch_size, total_len=self._total_len, shuffle=self._shuffle)
            
    def __getitem__(self, index):
        feature = jt.array(self._data[index][0])
        label = jt.argmax(jt.array(self._data[index][1]), dim = 0)[0]
        return feature, label
    
trainDataset = TrainDataset(vocab = train_vocab, data = train_data)


# In[6]:


class TextRNN(jt.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=1, num_emotion=7):
        super(TextRNN, self).__init__()
        self._num_hiddens = num_hiddens
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._embedding = nn.Embedding(vocab_size, embed_size)
        self._encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self._decoder = nn.Linear(4 * num_hiddens, num_emotion)
        
    def execute(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, len_sentence) 的整数张量
        @return:
            outputs: 对文本情感的预测，形状为 (batch_size, num_emotion) 的张量
        '''
        batch_size = inputs.shape[0]
        inputs = inputs.flatten()
        embeddings = self._embedding(inputs)
        # embeddings -> (batch_size, len_sentence, embed_size)
        embeddings = jt.reshape(embeddings, (batch_size, embeddings.shape[0]//batch_size, embeddings.shape[1]))
        embeddings = embeddings.permute(1, 0, 2)
        # embeddings -> (len_sentence, batch_size, embed_size)
        hx = (jt.zeros((self._num_layers*2, batch_size, self._num_hiddens)),
              jt.zeros((self._num_layers*2, batch_size, self._num_hiddens)))
        hiddens, _ = self._encoder(embeddings, hx)
        encoding = jt.contrib.concat([hiddens[0], hiddens[-1]], dim = -1)
        print(encoding.shape)
        outputs = self._decoder(encoding)
        return outputs

vocab_size = train_vocab.size()
embed_size, num_hiddens, num_layers = 300, 300, 2
net = TextRNN(vocab_size, embed_size, num_hiddens, num_layers)
print(len(net.parameters()))


# In[7]:


def train(model, train_loader, optimizer, init_lr, epoch, max_epoch):
    model.train()
    max_iter = len(train_loader)
    
    sum_tot = 0
    sum_acc = 0
    for index, (feature, label) in enumerate(train_loader):
        pred = model(feature)
        loss = nn.cross_entropy_loss(pred, label)
        optimizer.step(loss)
        if jt.argmax(pred, dim = -1)[0] == label[0]:
            sum_acc = sum_acc + 1
        sum_tot = sum_tot + 1
        if index % 500 == 0:
            print ('Training in epoch {} iteration {} acc = {} loss = {}'.format(epoch, index, sum_acc/sum_tot, loss.data[0]))

def evaluate(model, test_loader, epoch):
    model.eval()
    
    sum_acc = 0
    sum_tot = 0
    for index, (feature, label) in enumerate(test_loader):
        pred = model(feature)
        if jt.argmax(pred, dim = -1)[0] == label[0]:
            sum_acc = sum_acc + 1
        sum_tot = sum_tot + 1
    print ("Testing in epoch {}, acc = {}".format(epoch, sum_acc / sum_tot))


# In[8]:


test_ID_name = folder_name + "ID_test"
test_result_name = folder_name + "ISEAR_test"
test_data = decode_file(test_ID_name, test_result_name)
testDataset = TrainDataset(vocab = train_vocab, data = test_data)


learning_rate = 0.0001
max_epoch = 5
optimizer = nn.Adam(net.parameters(), learning_rate)
for epoch in range(max_epoch):
    train(net, trainDataset, optimizer, learning_rate, epoch, max_epoch)
    evaluate(net, testDataset, epoch)


# In[ ]:




