import os
import time

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import loss as gloss, nn, rnn
from sklearn.metrics import precision_recall_fscore_support, classification_report

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "cnssnn_pcnn_SemEval", "net_cnssnn_pcnn_epoch%d.params")
WORD_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = WORD_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60
MAX_ENTITY_DEGREE = 50
ENTITY_DEGREE = MAX_ENTITY_DEGREE + 1
MASK_LENGTH = ENTITY_DEGREE
ENTITY_EDGE_VEC_LENGTH = ENTITY_DEGREE * (WORD_DIMENSION * 2)
VEC_LENGTH = DIMENSION * FIXED_WORD_LENGTH + ENTITY_EDGE_VEC_LENGTH * 2
ADAPTIVE_LEARNING_RATE = False

CTX = mx.gpu(7)
ctx = [CTX]
fail_id_file = open("fail_id_cnssnn_pcnn_SemEval.txt", "w")

input_train = np.load('data_train_cnssnn_SemEval.npy')
input_test = np.load('data_test_cnssnn_SemEval.npy')

x_train = input_train[:, 2:]
y_train = input_train[:, 0:2]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, 2:]
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
epochs = 300
gap = 50

batch_size = 128
loss = gloss.SoftmaxCrossEntropyLoss()

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
        a, b = accuracy_with_flag(net(X), y[:, 0])
        acc += a
        for i in range(len(b)):
            if not b[i]:
                fail_id.append(str(int(y[i, 1].asscalar())))
    fail_id_file.write("%s\n" % " ".join(fail_id))
    return acc / len(data_iter)


def train(net, train_iter, test_iter):
    highest_epoch = -1
    highest_acc = -1
    for epoch in range(1, epochs + 1):
        train_loss_sum = 0
        train_acc_sum = 0
        start = time.time()
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
            print("learning_rate decay: %f" % trainer.learning_rate)
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                lss = loss(y_hat, y[:, 0])
            lss.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
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
    np.save("result_cnssnn_pcnn_SemEval.npy", result.asnumpy())
    predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
    label_list = y_test_0.astype(np.int).asnumpy().tolist()
    print(precision_recall_fscore_support(label_list, predict_list, average='macro'))
    print(classification_report(label_list, predict_list))


class Network(nn.Block):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(230, kernel_size=[3, DIMENSION], padding=[2, 0], strides=1)
            self.pmp = nn.MaxPool2D(pool_size=[FIXED_WORD_LENGTH, 1])
            self.conv_out = nn.Sequential()
            self.conv_out.add(nn.Flatten())
            self.conv_out.add(nn.Activation(activation='relu'))

            self.center_att = nn.Sequential()
            self.center_att.add(nn.Dense(1, in_units=200, flatten=False,
                                         activation="sigmoid"))
            self.center_out = nn.Sequential()
            # self.center_out.add(nn.Dense(200, activation="relu"))
            self.center_out.add(nn.Activation(activation='relu'))

            self.output = nn.Sequential()
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(128, activation="sigmoid"))
            self.output.add(nn.Dense(18, activation='tanh'))

    def forward(self, input_data):
        ep1 = input_data[:, 0].astype(int).asnumpy().tolist()
        ep2 = input_data[:, 1].astype(int).asnumpy().tolist()
        input_data = input_data[:, 2:]
        e1_vec_start = FIXED_WORD_LENGTH * DIMENSION
        x = input_data[:, :e1_vec_start].reshape(
            (input_data.shape[0], FIXED_WORD_LENGTH, DIMENSION))  # (m, 60, 110)
        x = nd.expand_dims(x, axis=1)

        e1neimask = input_data[:, e1_vec_start:e1_vec_start + MASK_LENGTH]  # (m, 51)
        e1edge = input_data[:, e1_vec_start + MASK_LENGTH:e1_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH].reshape(
            (input_data.shape[0], ENTITY_DEGREE, WORD_DIMENSION * 2))  # (m, 51, 200)
        e1neigh = e1edge[:, :, :WORD_DIMENSION]

        e2_vec_start = e1_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH
        e2neimask = input_data[:, e2_vec_start:e2_vec_start + MASK_LENGTH]  # (m, 51)
        e2edge = input_data[:, e2_vec_start + MASK_LENGTH:e2_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH].reshape(
            (input_data.shape[0], ENTITY_DEGREE, WORD_DIMENSION * 2))  # (m, 51,200)
        e2neigh = e2edge[:, :, :WORD_DIMENSION]

        # x.shape = (128, 1, 60, 110) NCHW
        conv_result = nd.relu(self.conv(x))  # (128, 230, 62, 1) NCHW
        be1_mask = nd.zeros(conv_result.shape, ctx=CTX)
        aes_mask = nd.zeros(conv_result.shape, ctx=CTX)
        be2_mask = nd.zeros(conv_result.shape, ctx=CTX)
        #
        be1_pad = nd.ones(conv_result.shape, ctx=CTX) * (-100)
        aes_pad = nd.ones(conv_result.shape, ctx=CTX) * (-100)
        be2_pad = nd.ones(conv_result.shape, ctx=CTX) * (-100)
        for i in range(x.shape[0]):
            if ep1[i] == 0:
                ep1[i] += 1
                ep2[i] += 1
            be1_mask[i, :, :ep1[i], :] = 1
            be1_pad[i, :, :ep1[i], :] = 0
            aes_mask[i, :, ep1[i]:ep2[i], :] = 1
            aes_pad[i, :, ep1[i]:ep2[i], :] = 0
            be2_mask[i, :, ep2[i]:, :] = 1
            be2_pad[i, :, ep2[i]:, :] = 0
        be1 = conv_result * be1_mask
        aes = conv_result * aes_mask
        be2 = conv_result * be2_mask
        be1 = be1 + be1_pad
        aes = aes + aes_pad
        be2 = be2 + be2_pad
        o1 = self.pmp(be1)
        o2 = self.pmp(aes)
        o3 = self.pmp(be2)
        out = nd.concat(o1, o2, o3, dim=2)  # (128, 230, 3, 1)
        y1 = self.conv_out(out)

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
        center_y = self.center_out(center_y)

        y4 = nd.concat(y1, center_y, dim=1)
        y5 = self.output(y4)
        return y5


net = Network()
net.initialize(ctx=ctx)
if ADAPTIVE_LEARNING_RATE:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'beta1': 0.9, 'beta2': 0.99, 'learning_rate': 1e-2})
else:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})
train(net, train_data, test_data)

fail_id_file.close()
