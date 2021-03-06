import os
import time

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd, init
from mxnet.gluon import loss as gloss, nn, rnn
from sklearn.metrics import precision_recall_fscore_support, classification_report

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "cnssnn_desc", "net_cnssnn_desc_epoch%d_12610.params")
WORD_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = WORD_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60
MAX_ENTITY_DEGREE = 50
ENTITY_DEGREE = MAX_ENTITY_DEGREE + 1
MASK_LENGTH = ENTITY_DEGREE
ENTITY_EDGE_VEC_LENGTH = ENTITY_DEGREE * (WORD_DIMENSION * 2)
VEC_LENGTH = DIMENSION * FIXED_WORD_LENGTH + ENTITY_EDGE_VEC_LENGTH * 2
ADAPTIVE_LEARNING_RATE = True
DESC_LOSS_RATE = 1
DESC_LENGTH = 80
CNSSNN_START_POINT = WORD_DIMENSION * 2 + DESC_LENGTH * WORD_DIMENSION * 2

CTX = mx.gpu(1)
ctx = [CTX]
fail_id_file = open("fail_id_cnssnn.txt", "w")

input_train = np.load('data_train_cnssnn_id_desc.npy')
input_test = np.load('data_test_cnssnn_id_desc.npy')

x_train = input_train[:, 4:]
y_train = input_train[:, 0:2]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, 4:]
y_test = input_test[:, 0:2]
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

decay_rate = 0.1
epochs = 100
gap = 50
batch_size = 128

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train, y_train), batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test, y_test), batch_size, shuffle=False)


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y).mean().asscalar()


def accuracy_with_flag(y_hat, y):
    return (y_hat.argmax(axis=1) == y).mean().asscalar(), (y_hat.argmax(axis=1) == y)


def evaluate_accuracy(data_iter, net):
    acc = 0
    fail_id = []
    for X, y in data_iter:
        a, b = accuracy_with_flag(net(X[:, CNSSNN_START_POINT:]), y[:, 0])
        acc += a
        for i in range(len(b)):
            if not b[i]:
                fail_id.append(str(int(y[i, 1].asscalar())))
    fail_id_file.write("%s\n" % " ".join(fail_id))
    return acc / len(data_iter)


def train(net, net2, train_iter, test_iter):
    highest_epoch = -1
    highest_acc = -1

    for epoch in range(1, epochs + 1):
        train_loss_sum = 0
        train_acc_sum = 0
        start = time.time()
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0 and trainer.learning_rate > 0.0001:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
            print("learning_rate decay: %f" % trainer.learning_rate)
        for X, y in train_iter:
            with autograd.record():
                en1 = X[:, 0:WORD_DIMENSION]
                en2 = X[:, WORD_DIMENSION:WORD_DIMENSION * 2]
                en1_desc = X[:, WORD_DIMENSION * 2:
                                WORD_DIMENSION * 2 + DESC_LENGTH * WORD_DIMENSION] \
                    .reshape((X.shape[0], DESC_LENGTH, WORD_DIMENSION)) \
                    .expand_dims(axis=1)
                en2_desc = X[:, WORD_DIMENSION * 2 + DESC_LENGTH * WORD_DIMENSION:
                                WORD_DIMENSION * 2 + DESC_LENGTH * WORD_DIMENSION * 2] \
                    .reshape((X.shape[0], DESC_LENGTH, WORD_DIMENSION)) \
                    .expand_dims(axis=1)

                y_hat = net(X[:, CNSSNN_START_POINT:])
                y_hat2 = net2(en1_desc)
                y_hat3 = net2(en2_desc)
                lss = loss(y_hat, y[:, 0]) + DESC_LOSS_RATE * (loss2(y_hat2, en1) + loss2(y_hat3, en2))
            lss.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
            trainer2.step(batch_size)
            train_loss_sum += lss.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y[:, 0])
        test_acc = evaluate_accuracy(test_iter, net)
        if test_acc > highest_acc:
            highest_acc = test_acc
            highest_epoch = epoch
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f time %.1f sec'
              % (epoch, train_loss_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc, time.time() - start))
        net.save_params(SAVE_MODEL_PATH % epoch)
    print("highest epoch & acc: %d, %f" % (highest_epoch, highest_acc))
    evaluate_model(net, highest_epoch)


def evaluate_model(net, epoch):
    net.load_params(SAVE_MODEL_PATH % epoch, ctx=CTX)
    y_hat = net(x_test)
    y_test_0 = y_test[:, 0]
    result = nd.concat(y_test_0.expand_dims(axis=1), y_hat, dim=1)
    np.save("result_cnssnn_desc.npy", result.asnumpy())
    predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
    label_list = y_test_0.astype(np.int).asnumpy().tolist()
    print(precision_recall_fscore_support(label_list, predict_list, average='weighted'))
    print(classification_report(label_list, predict_list))


class Network(nn.Block):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)
        with self.name_scope():
            self.gru = rnn.GRU(100, num_layers=1, bidirectional=True)
            self.gru_out = nn.Sequential()
            self.gru_out.add(nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1)), )
            self.gru_out.add(nn.Flatten())
            self.gru_out.add(nn.Activation(activation='relu'))

            self.center_att = nn.Sequential()
            self.center_att.add(nn.Dense(1, in_units=200, flatten=False,
                                         activation="sigmoid"))
            self.center_out = nn.Sequential()
            self.center_out.add(nn.Dense(200, activation="relu"))
            self.output = nn.Sequential()
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(7))

    def forward(self, input_data):
        e1_vec_start = FIXED_WORD_LENGTH * DIMENSION
        x = input_data[:, :e1_vec_start].reshape(
            (input_data.shape[0], FIXED_WORD_LENGTH, DIMENSION))  # (m, 60, 110)
        e1neimask = input_data[:, e1_vec_start:e1_vec_start + MASK_LENGTH]  # (m, 51)
        e1edge = input_data[:, e1_vec_start + MASK_LENGTH:e1_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH].reshape(
            (input_data.shape[0], ENTITY_DEGREE, WORD_DIMENSION * 2))  # (m, 51, 200)
        e1neigh = e1edge[:, :, :WORD_DIMENSION]

        e2_vec_start = e1_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH
        e2neimask = input_data[:, e2_vec_start:e2_vec_start + MASK_LENGTH]  # (m, 51)
        e2edge = input_data[:, e2_vec_start + MASK_LENGTH:e2_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH].reshape(
            (input_data.shape[0], ENTITY_DEGREE, WORD_DIMENSION * 2))  # (m, 51,200)
        e2neigh = e2edge[:, :, :WORD_DIMENSION]

        gru = self.gru
        x = nd.transpose(x, axes=(1, 0, 2))
        h = gru(x)
        ht = nd.transpose(h, axes=(1, 0, 2))
        gru_out = self.gru_out
        y1 = gru_out(ht.expand_dims(1))  # (m,200)

        att = self.center_att
        e1edge = nd.tanh(e1edge)
        e1g = att(e1edge)  # (m,51,1)
        e1g = e1g * e1neimask.expand_dims(2)
        e1g = nd.softmax(e1g, axis=1)
        e1gt = nd.transpose(e1g, axes=(0, 2, 1))  # (m,1,151)
        e1n = nd.batch_dot(e1gt, e1neigh)  # (m,1,100)
        e1n = e1n.reshape((e1n.shape[0], 100))  # (m,100)

        e2edge = nd.tanh(e2edge)
        e2g = att(e2edge)  # (m,51,1)
        e2g = e2g * e2neimask.expand_dims(2)
        e2g = nd.softmax(e2g, axis=1)
        e2gt = nd.transpose(e2g, axes=(0, 2, 1))  # (m,1,151)
        e2n = nd.batch_dot(e2gt, e2neigh)  # (m,1,100)
        e2n = e2n.reshape((e2n.shape[0], 100))  # (m,100)

        center_y = nd.concat(e1n, e2n, dim=1)  # (m,200)
        center_out = self.center_out
        center_y = center_out(center_y)

        out = self.output
        y4 = nd.concat(y1, center_y, dim=1)
        y5 = out(y4)
        return y5


net = Network()
net.collect_params().initialize(init=init.Xavier(), ctx=ctx)
if ADAPTIVE_LEARNING_RATE:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'beta1': 0.9, 'beta2': 0.99, 'learning_rate': 1e-2})
else:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})

net2 = nn.Sequential()
with net2.name_scope():
    net2.add(nn.Conv2D(256, kernel_size=(3, WORD_DIMENSION), padding=(1, 0), activation='relu'))
    net2.add(nn.MaxPool2D(pool_size=(DESC_LENGTH, 1)))
    net2.add(nn.Dense(WORD_DIMENSION))
net2.collect_params().initialize(init=init.Xavier(), ctx=ctx)
print(net2)

loss = gloss.SoftmaxCrossEntropyLoss()
loss2 = gloss.L2Loss()
trainer2 = gluon.Trainer(net2.collect_params(), 'adam', {'learning_rate': 0.0001})

train(net, net2, train_data, test_data)

fail_id_file.close()
