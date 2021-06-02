#!/usr/bin/env python
# coding: utf-8

import random
import jittor as jt
from jittor import nn


class TextRNN(jt.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=1, num_emotion=2):
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

def train(model, optimizer):
    model.train()
    
    feature = jt.randn((1, 180))
    label = jt.zeros((1, 7))
    pred = model(feature)
    loss = nn.cross_entropy_loss(pred, label)
    optimizer.step(loss)

vocab_size, embed_size, num_hiddens, num_layers, num_emotions = 180, 300, 300, 2, 7
net = TextRNN(vocab_size, embed_size, num_hiddens, num_layers, num_emotions)

learning_rate = 0.0001
optimizer = nn.Adam(net.parameters(), learning_rate)
train(net, optimizer)
