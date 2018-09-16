import time
import os
import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn, rnn
from sklearn.metrics import precision_recall_fscore_support, classification_report

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "lstm_SemEval", "net_lstm_epoch%d.params")
SENTENCE_DIMENSION = 100
DIMENSION = SENTENCE_DIMENSION
FIXED_WORD_LENGTH = 60
ADAPTIVE_LEARNING_RATE = True

CTX = mx.gpu(6)
ctx = [CTX]

input_train = np.load('data_train_rnn_SemEval.npy')
input_test = np.load('data_test_rnn_SemEval.npy')
x_train = input_train[:, 1:].reshape((input_train.shape[0], FIXED_WORD_LENGTH, DIMENSION))
# x_train = np.expand_dims(x_train, axis=1)
y_train = input_train[:, 0]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, 1:].reshape((input_test.shape[0], FIXED_WORD_LENGTH, DIMENSION))
# x_test = np.expand_dims(x_test, axis=1)
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
            self.lstm = rnn.LSTM(64, num_layers=1, bidirectional=True, layout="NTC", dropout=0.3)
            self.output = nn.Sequential()
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(9))

    def forward(self, input_data):
        h = self.lstm(input_data)
        return self.output(h)


net = Network()
net.collect_params().initialize(init=init.Xavier(), ctx=ctx)
print(net)

batch_size = 100
num_epochs = 120
decay_rate = 0.1
gap = 25
loss = gloss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'AdaDelta', {'rho': 0.95, 'epsilon': 1e-6, 'wd': 0.01})
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
if ADAPTIVE_LEARNING_RATE:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01, 'wd': 1e-5})
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
        train_l_sum = 0
        train_acc_sum = 0
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0 and trainer.learning_rate > 0.0001:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
            print("learning_rate decay: %f" % trainer.learning_rate)
        start = time.time()
        for X, y in train_iter:
            y = y.copyto(CTX)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        if test_acc > highest_acc:
            highest_acc = test_acc
            highest_epoch = epoch
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f time %.1f sec'
              % (epoch, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc, time.time() - start))
        net.save_params(SAVE_MODEL_PATH % epoch)
    print("highest epoch & acc: %d, %f" % (highest_epoch, highest_acc))
    evaluate_model(net, highest_epoch)


def evaluate_model(net, epoch):
    net.load_params(SAVE_MODEL_PATH % epoch, ctx=CTX)
    y_hat = net(x_test)
    result = nd.concat(y_test.expand_dims(axis=1), y_hat, dim=1)
    np.save("result_lstm_SemEval.npy", result.asnumpy())
    predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
    label_list = y_test.astype(np.int).asnumpy().tolist()
    print(precision_recall_fscore_support(label_list, predict_list, average='macro'))
    print(classification_report(label_list, predict_list))


train(net, train_data, test_data, loss, num_epochs, batch_size, trainer)
