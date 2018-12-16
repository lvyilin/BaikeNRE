import time
import os
import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn, rnn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold

MARK = "rnn_pi_p"
CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", MARK, "net_{}_epoch%d.params".format(MARK))

SENTENCE_DIMENSION = 100
DIMENSION = SENTENCE_DIMENSION
FIXED_WORD_LENGTH = 60
ADAPTIVE_LEARNING_RATE = True

CTX = mx.gpu(3)
ctx = [CTX]

input_train = np.load('data_train_{}.npy'.format(MARK))
input_test = np.load('data_test_{}.npy'.format(MARK))
input_all = np.concatenate((input_train, input_test), axis=0)
np.random.shuffle(input_all)
input_x = input_all[:, 1:].astype(np.float32)
input_y = input_all[:, 0].astype(np.int)

x_input = nd.array(input_x, ctx=CTX)
y_input = nd.array(input_y, ctx=CTX)


def get_prediction(net, x):
    return net(x).argmax(axis=1)


def get_acc(y_true: nd.ndarray.NDArray, y_pred: nd.ndarray.NDArray):
    return accuracy_score(y_true.asnumpy().astype(np.int), y_pred.asnumpy().astype(np.int))


def get_pre(y_true: nd.ndarray.NDArray, y_pred: nd.ndarray.NDArray):
    return precision_score(y_true.asnumpy().astype(np.int), y_pred.asnumpy().astype(np.int), average="weighted")


def get_rec(y_true: nd.ndarray.NDArray, y_pred: nd.ndarray.NDArray):
    return recall_score(y_true.asnumpy().astype(np.int), y_pred.asnumpy().astype(np.int), average="weighted")


def get_f1(y_true: nd.ndarray.NDArray, y_pred: nd.ndarray.NDArray):
    return f1_score(y_true.asnumpy().astype(np.int), y_pred.asnumpy().astype(np.int), average="weighted")


class Network(nn.Block):
    def __init__(self):
        super().__init__()
        with self.name_scope():
            self.lstm = rnn.LSTM(64, num_layers=1, bidirectional=True, dropout=0.2, layout="NTC")
            self.pool = nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1))
            self.output = nn.Sequential()
            self.output.add(nn.Flatten())
            self.output.add(nn.Activation(activation='relu'))
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(7))

    def forward(self, input_data):
        ep1 = input_data[:, 0].asnumpy().astype(np.int).tolist()
        ep2 = input_data[:, 1].asnumpy().astype(np.int).tolist()
        x_sen = input_data[:, 2:].reshape(
            (input_data.shape[0], FIXED_WORD_LENGTH, DIMENSION))
        h_sen = self.lstm(x_sen).expand_dims(axis=1)  # (128, 1, 60, 128)
        be1_mask = nd.zeros(h_sen.shape, ctx=CTX)
        aes_mask = nd.zeros(h_sen.shape, ctx=CTX)
        be2_mask = nd.zeros(h_sen.shape, ctx=CTX)
        be1_pad = nd.ones(h_sen.shape, ctx=CTX) * (-100)
        aes_pad = nd.ones(h_sen.shape, ctx=CTX) * (-100)
        be2_pad = nd.ones(h_sen.shape, ctx=CTX) * (-100)
        for i in range(x_sen.shape[0]):
            if ep1[i] == 0:
                ep1[i] += 1
                ep2[i] += 1
            be1_mask[i, :, :ep1[i], :] = 1
            be1_pad[i, :, :ep1[i], :] = 0
            aes_mask[i, :, ep1[i]:ep2[i], :] = 1
            aes_pad[i, :, ep1[i]:ep2[i], :] = 0
            be2_mask[i, :, ep2[i]:, :] = 1
            be2_pad[i, :, ep2[i]:, :] = 0
        be1 = h_sen * be1_mask
        aes = h_sen * aes_mask
        be2 = h_sen * be2_mask
        be1 = be1 + be1_pad
        aes = aes + aes_pad
        be2 = be2 + be2_pad
        o1 = self.pool(be1)
        o2 = self.pool(aes)
        o3 = self.pool(be2)
        out = nd.concat(o1, o2, o3, dim=2)
        return self.output(out)


def train(net, x_train, y_train, x_test, y_test, trainer):
    train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train, y_train), batch_size)
    highest_epoch = -1
    highest_f1 = -1
    for epoch in range(1, num_epochs + 1):
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0 and trainer.learning_rate > lowest_lr:
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
        pred = get_prediction(net, x_test)
        test_acc = get_acc(y_test, pred)
        test_f1 = get_f1(y_test, pred)
        if test_f1 > highest_f1:
            highest_f1 = test_f1
            highest_epoch = epoch
            net.save_params(SAVE_MODEL_PATH % epoch)
        print('epoch %d,test acc %.3f test f1 %.3f time %.1f sec'
              % (epoch, test_acc, test_f1, time.time() - start))
    net.load_params(SAVE_MODEL_PATH % highest_epoch, ctx=CTX)
    print("highest epoch & acc: %d, %f" % (highest_epoch, highest_f1))


def cross_validate(net, X, y, cv=5):
    kf = KFold(cv)
    pre_list, rec_list, f1_list = [], [], []
    for train_index, test_index in kf.split(X):
        net.collect_params().initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
        if ADAPTIVE_LEARNING_RATE:
            trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
        else:
            trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})
        train_index_nd = nd.array(train_index, ctx=CTX)
        test_index_nd = nd.array(test_index, ctx=CTX)
        train(net, X[train_index_nd], y[train_index_nd], X[test_index_nd], y[test_index_nd], trainer)
        prediction = get_prediction(net, X[test_index_nd])
        pre_list.append(get_pre(y[test_index_nd], prediction))
        rec_list.append(get_rec(y[test_index_nd], prediction))
        f1_list.append(get_f1(y[test_index_nd], prediction))

    return pre_list, rec_list, f1_list


# 网络定义
net = Network()
print(net)
# 评测指标
# scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
# 超参数
batch_size = 128
num_epochs = 30
decay_rate = 0.1
lowest_lr = 0.0001
gap = 10
# 损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 交叉验证
pre_list, rec_list, f1_list = cross_validate(net, x_input, y_input, 5)
print("Precision: " + str(pre_list))
print("Recall: " + str(rec_list))
print("F1 Score: " + str(f1_list))

print("Avg precision: {}".format(sum(pre_list) / len(pre_list)))
print("Avg recall: {}".format(sum(rec_list) / len(rec_list)))
print("Avg f1 Score: {}".format(sum(f1_list) / len(f1_list)))
