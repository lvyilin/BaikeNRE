import os
import time

import numpy as np
from mxnet import gluon, init, autograd
from mxnet.gluon import loss as gloss, nn

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "cnn", "net_cnn_epoch%d.params")
SENTENCE_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = SENTENCE_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60
ADAPTIVE_LEARNING_RATE = False

input_train = np.load('data_train.npy')
input_test = np.load('data_test.npy')
x_train = input_train[:, 3:].reshape((input_train.shape[0], FIXED_WORD_LENGTH, DIMENSION))
x_train = np.expand_dims(x_train, axis=1)
y_train = input_train[:, 0]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, 3:].reshape((input_test.shape[0], FIXED_WORD_LENGTH, DIMENSION))
x_test = np.expand_dims(x_test, axis=1)
y_test = input_test[:, 0]
print(x_test.shape)
print(y_test.shape)
print(y_train[0:5])
print(y_test[0:5])

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
print(x_train.shape, x_test.shape)

net = nn.Sequential()
# Simple DNN
# net.add(nn.Dense(64, activation='relu'))
# net.add(nn.Dense(32, activation='relu'))
# net.add(nn.Dense(10))
# net.initialize(ctx=ctx, init=init.Xavier())

# CNN, CR-CNN
# net.add(nn.Conv2D(256, kernel_size=(5, DIMENSION), padding=(1, 0), activation='relu'))
net.add(nn.Conv2D(256, kernel_size=(3, DIMENSION), padding=(1, 0), activation='relu'))
# net.add(nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH - 2, 1)))

net.add(nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1)))
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dropout(0.5))
net.add(nn.Dense(11))

net.initialize(init=init.Xavier())

print(net)

batch_size = 128
num_epochs = 100
decay_rate = 0.1
gap = 25
loss = gloss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'AdaDelta', {'rho': 0.95, 'epsilon': 1e-6, 'wd': 0.01})
# trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
if ADAPTIVE_LEARNING_RATE:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
else:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train, y_train), batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test, y_test), batch_size, shuffle=False)


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


def train(net, train_iter, test_iter, loss, num_epochs, batch_size, trainer):
    for epoch in range(1, num_epochs + 1):
        train_loss_sum = 0
        train_acc_sum = 0
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
            print("learning_rate decay: %f" % trainer.learning_rate)
        start = time.time()
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                lss = loss(y_hat, y)
            lss.backward()
            trainer.step(batch_size)
            train_loss_sum += lss.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f time %.1f sec'
              % (epoch, train_loss_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc, time.time() - start))
        net.save_parameters(SAVE_MODEL_PATH % epoch)


train(net, train_data, test_data, loss, num_epochs, batch_size, trainer)
