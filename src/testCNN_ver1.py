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


# In[22]:


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

train_vocab = build_vocab(train_data)


# In[28]:


class TrainDataset(Dataset):
    def __init__(self, vocab=None, data=None, batch_size=1, shuffle=False):
        super().__init__()
        
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._total_len = len(data)
        self._data = []
        for sentence in data:
            self._data.append([[vocab.locate(x) for x in sentence[0]], sentence[1]])
        self.set_attrs(batch_size=self._batch_size, total_len=self._total_len, shuffle=self._shuffle)
            
    def __getitem(self, index):
        feature = jt.array(self._data[index][0])
        label = jt.array(self._data[index][1])
        return feature, label
    
trainDataset = TrainDataset(vocab = train_vocab, data = train_data)


# In[ ]:




