import os
import time, json

import numpy as np
from gensim.models import KeyedVectors
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn, data as gdata

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
DIMENSION = 100
DESC_LENGTH = 80
CTX = mx.cpu(0)
ctx = [CTX]
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
PLACEHOLDER = np.zeros(DIMENSION)
desc_key = []
descvec_value = np.load("desc2vec_value.npy")

with open("desc2vec_key.txt", "r", encoding="utf8") as f:
    for line in f:
        desc_key.append(line.strip())
descvec_key = [wordvec[k] for k in desc_key]

X = nd.array(np.array(descvec_value), ctx=CTX).expand_dims(axis=1)
Y = nd.array(np.array(descvec_key), ctx=CTX).expand_dims(axis=1)
# print(X)
# print(Y)
num_epochs = 1
batch_size = len(X)
dataset = gdata.ArrayDataset(X, Y)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=False)

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(256, kernel_size=(3, DIMENSION), padding=(1, 0), activation='relu'))
    net.add(nn.MaxPool2D(pool_size=(DESC_LENGTH, 1)))
    net.add(nn.Dense(DIMENSION))
net.collect_params().initialize(init=init.Xavier(), ctx=ctx)
print(net)
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})


def evaluate_loss(net):
    data_iter = gdata.DataLoader(dataset, 1, shuffle=False)
    losses = []
    for X, y in data_iter:
        y_hat = net(X)
        lss = loss(y_hat, y)
        losses.append(lss.mean().asscalar().item())
    return losses


def train(net, data_iter, loss, num_epochs, batch_size, trainer):
    for epoch in range(1, num_epochs + 1):
        loss_sum = 0
        for X, y in data_iter:
            # y = y.copyto(CTX)
            with autograd.record():
                y_hat = net(X)
                lss = loss(y_hat, y)
            lss.backward()
            trainer.step(batch_size)
            loss_sum += lss.mean().asscalar()
        print(loss_sum / len(data_iter))
    result = evaluate_loss(net)
    result_dict = dict(zip(desc_key, result))
    with open("desc_losses.json", "w", encoding="utf8") as fp:
        json.dump(result_dict, fp, ensure_ascii=False)


train(net, data_iter, loss, num_epochs, batch_size, trainer)
