import os
import time

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd, init
from mxnet.gluon import loss as gloss, nn, rnn, utils as gutils
from sklearn.metrics import precision_recall_fscore_support, classification_report

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "lstm_infobox", "net_lstm_infobox_epoch%d.params")
WORD_DIMENSION = 100
DIMENSION = WORD_DIMENSION
FIXED_WORD_LENGTH = 60
INFOBOX_VALUE_LENGTH = 10
INFOBOX_LENGTH = 20
ADAPTIVE_LEARNING_RATE = True

ctx = [mx.gpu(i) for i in (1, 2, 4, 5, 6, 7)]
CTX = ctx[0]

input_train = np.load('data_train_rnn_infobox.npy')
input_test = np.load('data_test_rnn_infobox.npy')

x_train = input_train[:, 1:]
y_train = input_train[:, 0]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, 1:]
y_test = input_test[:, 0]
print(x_test.shape)
print(y_test.shape)

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
print(x_train.shape, x_test.shape)

x_train = nd.array(x_train, ctx=CTX)
y_train = nd.array(y_train, ctx=CTX)
x_test = nd.array(x_test, ctx=CTX)
y_test = nd.array(y_test, ctx=CTX)


class Network(nn.Block):
    def __init__(self, prefix=None, params=None):
        super().__init__(prefix, params)
        with self.name_scope():
            self.lstm = rnn.LSTM(64, num_layers=1, bidirectional=True, dropout=0.2, layout='NTC')
            self.lstm_out = nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1))
            self.infobox_lstm = rnn.LSTM(64, num_layers=1, bidirectional=True, layout='NTC')
            self.infobox_layer = nn.Sequential()
            self.infobox_layer.add(nn.MaxPool2D(pool_size=(INFOBOX_VALUE_LENGTH, 1)))
            self.att = nn.Sequential()
            self.att.add(nn.Dense(128, flatten=False,
                                  activation="sigmoid"))
            self.output = nn.Sequential()
            self.output.add(nn.Flatten())
            self.output.add(nn.Activation(activation='relu'))
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(7))

    def forward(self, input_data: mx.ndarray.ndarray.NDArray):
        context = input_data.context

        x_sen = input_data[:, :DIMENSION * FIXED_WORD_LENGTH].reshape(
            (input_data.shape[0], FIXED_WORD_LENGTH, DIMENSION))
        e1_start = DIMENSION * FIXED_WORD_LENGTH
        e1_infobox = input_data[:, e1_start:e1_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION].reshape(
            (input_data.shape[0], INFOBOX_LENGTH, INFOBOX_VALUE_LENGTH, DIMENSION))
        e2_start = e1_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION
        e2_infobox = input_data[:, e2_start:e2_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION].reshape(
            (input_data.shape[0], INFOBOX_LENGTH, INFOBOX_VALUE_LENGTH, DIMENSION))

        h_sen = self.lstm(x_sen)  # (128, 60, hidden_size*2)
        ht_sen = self.lstm_out(h_sen.expand_dims(1))  # (128, 1, hidden_size*2)
        ht_sen = ht_sen.reshape((ht_sen.shape[0], ht_sen.shape[1], ht_sen.shape[3]))
        ht_sen = nd.transpose(ht_sen, axes=(0, 2, 1))

        e1_arr = nd.zeros((e1_infobox.shape[0], e1_infobox.shape[1], e1_infobox.shape[2], 64 * 2),
                          ctx=context)  # (128, 20, 10, 128)
        for i in range(e1_infobox.shape[0]):
            e1_arr[i] = self.infobox_lstm(e1_infobox[i])
        e1_t = self.infobox_layer(e1_arr)  # (128, 20, 1, 128)
        e1_t = e1_t.reshape((e1_t.shape[0], e1_t.shape[1], e1_t.shape[3]))  # (128, 20, 128)
        e1_dense_out = self.att(e1_t)  # (128, 20, 128)

        e1_alpha = nd.softmax(nd.tanh(nd.batch_dot(e1_dense_out, ht_sen)))  # (128, 20, 1)
        e1_alpha_t = nd.transpose(e1_alpha, axes=(0, 2, 1))
        e1 = nd.batch_dot(e1_alpha_t, e1_t)
        e1 = e1.reshape((e1.shape[0], -1))

        e2_arr = nd.zeros((e2_infobox.shape[0], e2_infobox.shape[1], e2_infobox.shape[2], 64 * 2),
                          ctx=context)  # (128, 20, 10, 128)
        for i in range(e2_infobox.shape[0]):
            e2_arr[i] = self.infobox_lstm(e2_infobox[i])
        e2_t = self.infobox_layer(e2_arr)  # (128, 20, 1, 128)
        e2_t = e2_t.reshape((e2_t.shape[0], e2_t.shape[1], e2_t.shape[3]))  # (128, 20, 128)
        e2_dense_out = self.att(e2_t)  # (128, 20, 128)

        e2_alpha = nd.softmax(nd.tanh(nd.batch_dot(e2_dense_out, ht_sen)))  # (128, 20, 1)
        e2_alpha_t = nd.transpose(e2_alpha, axes=(0, 2, 1))
        e2 = nd.batch_dot(e2_alpha_t, e2_t)
        e2 = e2.reshape((e2.shape[0], -1))

        y = nd.concat(ht_sen.reshape((ht_sen.shape[0], -1)), e1, e2, dim=1)
        out = self.output(y)
        return out


net = Network()
net.collect_params().initialize(init=init.Xavier(), ctx=ctx)
print(net)

batch_size = 128
num_epochs = 50
decay_rate = 0.1
gap = 25
loss = gloss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'AdaDelta', {'rho': 0.95, 'epsilon': 1e-6, 'wd': 0.01})
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
        y = y.copyto(CTX)
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


def train(net, train_iter, test_iter, loss, num_epochs, batch_size, trainer):
    highest_epoch = -1
    highest_acc = -1
    for epoch in range(1, num_epochs + 1):
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
            print("learning_rate decay: %f" % trainer.learning_rate)
        start = time.time()
        for X, y in train_iter:
            # y = y.copyto(CTX)
            gpu_Xs = gutils.split_and_load(X, ctx, even_split=False)
            gpu_ys = gutils.split_and_load(y, ctx, even_split=False)
            with autograd.record():
                losses = [loss(net(gpu_X), gpu_y)
                          for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in losses:
                l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
        nd.waitall()

        test_acc = evaluate_accuracy(test_iter, net)
        if test_acc > highest_acc:
            highest_acc = test_acc
            highest_epoch = epoch
        print('epoch %d, test acc %.3f time %.1f sec'
              % (epoch, test_acc, time.time() - start))
        net.save_params(SAVE_MODEL_PATH % epoch)
    print("highest epoch & acc: %d, %f" % (highest_epoch, highest_acc))
    evaluate_model(net, highest_epoch)


def evaluate_model(net, epoch):
    net.load_params(SAVE_MODEL_PATH % epoch, ctx=CTX)
    y_hat = net(x_test)
    result = nd.concat(y_test.expand_dims(axis=1), y_hat, dim=1)
    np.save("result_lstm_infobox.npy", result.asnumpy())
    predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
    label_list = y_test.astype(np.int).asnumpy().tolist()
    print(precision_recall_fscore_support(label_list, predict_list, average='weighted'))
    print(classification_report(label_list, predict_list))


train(net, train_data, test_data, loss, num_epochs, batch_size, trainer)
