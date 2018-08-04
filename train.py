import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn

DIM = 100
FIXED_WORD_LENGTH = 78
input_train = np.load('data_train.npy')
input_test = np.load('data_test.npy')
x_train = input_train[:, :-1].reshape((input_train.shape[0], FIXED_WORD_LENGTH, DIM))
y_train = input_train[:, -1]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, :-1].reshape((input_test.shape[0], FIXED_WORD_LENGTH, DIM))
y_test = input_test[:, -1]
print(x_test.shape)
print(y_test.shape)

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
x_train = nd.array(x_train).as_in_context(mx.gpu(0))
y_train = nd.array(y_train).as_in_context(mx.gpu(0))
x_test = nd.array(x_test).as_in_context(mx.gpu(0))
y_test = nd.array(y_test).as_in_context(mx.gpu(0))
print(y_train.context, y_test.context)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(128, activation='relu'))
net.add(nn.Dense(10))
net.initialize(ctx=mx.gpu(0), init=init.Xavier())

batch_size = 128
num_epochs = 50
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'AdaDelta',
                        {'rho': 0.95, 'epsilon': 1e-6, 'wd': 0.001})
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

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
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                lss = loss(y_hat, y)
            lss.backward()
            trainer.step(batch_size)
            train_loss_sum += lss.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch, train_loss_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))


train(net, train_data, test_data, loss, num_epochs, batch_size, trainer)
