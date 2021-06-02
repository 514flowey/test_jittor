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
    file_ID = open(file_ID_name, "r")
    file_result = open(file_result_name, "r")
    data = []
    
    for line1, line2 in zip(file_ID, file_result):
        feature = line1.replace('\n','').lower().split(' ')
        label = line2.replace('\n','').lower().split(',')
        label = [int(x) for x in label]
        data.append([feature, label[1:]])
    
    random.shuffle(data)
    file_ID.close()
    file_result.close()
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


class Maxpool1D(jt.Module):
    def __init__(self):
        super(Maxpool1D, self).__init__()
    
    def execute(self, x):
        return x.max(dim=-1)


# In[7]:


class TextCNN(jt.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, num_emotion = 7):
        super(TextCNN, self).__init__()
        self._embedding = nn.Embedding(vocab_size, embed_size)
        self._constant_embedding = nn.Embedding(vocab_size, embed_size)
        self._pool = Maxpool1D()
        self._convs = nn.Sequential()
        for k, c in zip(kernel_sizes, num_channels):
            self._convs.append(nn.Conv1d(in_channels = 2 * embed_size,
                                       out_channels = c,
                                       kernel_size = k,))
        self._decoder = nn.Linear(sum(num_channels), num_emotion)
        self._dropout = nn.Dropout(0.5)
        
    def execute(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, len_sentence) 的整数张量
        @return:
            outputs: 对文本情感的预测，形状为 (batch_size, num_emotion) 的张量
        '''
        batch_size = inputs.shape[0]
        inputs = inputs.flatten()
        embeddings = jt.contrib.concat([self._embedding(inputs),
                                       self._constant_embedding(inputs)], dim = -1)
        # embeddings -> (batch_size, len_sentence, embed_size*2)
        embeddings = jt.reshape(embeddings, (batch_size, embeddings.shape[0]//batch_size, embeddings.shape[1]))
        embeddings = embeddings.permute(0, 2, 1)
        encoding = jt.contrib.concat([
            self._pool(nn.relu(conv(embeddings))) for conv in self._convs], dim = -1)
        outputs = self._decoder(self._dropout(encoding))
        return outputs
    
vocab_size = train_vocab.size()
embed_size, kernel_sizes, num_channels = 300, [2, 3, 4, 5], [64, 64, 64, 64]
net = TextCNN(vocab_size, embed_size, kernel_sizes, num_channels)


# In[8]:


# lr scheduler
def poly_lr_scheduler(opt, init_lr, index, epoch, max_iter, max_epoch):
    new_lr = init_lr * (1 - float(epoch * max_iter + index) / (max_epoch * max_iter)) ** 0.9
    opt.lr = new_lr

def train(model, train_loader, optimizer, init_lr, epoch, max_epoch):
    model.train()
    max_iter = len(train_loader)
    
    sum_tot = 0
    sum_acc = 0
    '''for parameter in model.parameters():
        print(parameter.shape)'''
    for index, (feature, label) in enumerate(train_loader):
#        poly_lr_scheduler(optimizer, init_lr, index, epoch, max_iter, max_epoch)
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


# In[9]:


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




    


# 
