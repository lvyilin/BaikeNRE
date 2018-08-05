import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn

DIM = 250
FIXED_WORD_LENGTH = 78

d_c = 256  # kernel_units
k = 3
r = np.sqrt(6 / (9 + d_c))

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

# ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
# # ctx = mx.gpu()
# x_train = nd.array(x_train).as_in_context(ctx)
# y_train = nd.array(y_train).as_in_context(ctx)
# x_test = nd.array(x_test).as_in_context(ctx)
# y_test = nd.array(y_test).as_in_context(ctx)
# print(y_train.context, y_test.context)

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)
print(x_train.shape, x_test.shape)

net = nn.Sequential()
# Simple DNN
# net.add(nn.Dense(64, activation='relu'))
# net.add(nn.Dense(32, activation='relu'))
# net.add(nn.Dense(10))
# net.initialize(ctx=ctx, init=init.Xavier())

# CNN
net.add(nn.Conv2D(d_c, kernel_size=(k, DIM), padding=(1, 0), activation='relu'))
net.add(nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1)))
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10, activation="relu"))

net.initialize(init=init.Xavier())
# 初始化全链接层参数，使用均匀分布
# net[2].weight.initialize(init=init.Uniform(scale=r), force_reinit=True)

print(net)

batch_size = 128
num_epochs = 50
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'AdaDelta',
                        {'rho': 0.95, 'epsilon': 1e-6, 'wd': 0.001})

# trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.1})
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
