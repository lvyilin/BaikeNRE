import os
import time

import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn
from sklearn.metrics import precision_recall_fscore_support, classification_report

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "pcnn_SemEval", "net_pcnn_epoch%d_12610.params")
SENTENCE_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = SENTENCE_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60
ADAPTIVE_LEARNING_RATE = True

CTX = mx.gpu(4)
ctx = [CTX]

input_train = np.load('data_train_cnn_SemEval.npy')
input_test = np.load('data_test_cnn_SemEval.npy')
# x_train = input_train[:, 3:].reshape((input_train.shape[0], FIXED_WORD_LENGTH, DIMENSION))
x_train = input_train[:, 1:]
y_train = input_train[:, 0]
print(x_train.shape)
print(y_train.shape)
# x_test = input_test[:, 3:].reshape((input_test.shape[0], FIXED_WORD_LENGTH, DIMENSION))
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


class PCNN(nn.Block):
    def __init__(self, **kwargs):
        super(PCNN, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(230, kernel_size=[3, DIMENSION], padding=[2, 0], strides=1)
            self.pmp = nn.MaxPool2D(pool_size=[FIXED_WORD_LENGTH, 1])

            self.output = nn.Sequential()
            self.output.add(nn.Flatten())
            self.output.add(nn.Activation(activation='tanh'))
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(128, activation='sigmoid'))
            self.output.add(nn.Dense(9, activation='tanh'))

    def forward(self, x):
        ep1 = x[:, 0].astype(int).asnumpy().tolist()
        ep2 = x[:, 1].astype(int).asnumpy().tolist()
        x = x[:, 2:].reshape((x.shape[0], FIXED_WORD_LENGTH, DIMENSION))
        x = nd.expand_dims(x, axis=1)
        # PCNN
        # 卷积结果
        conv_result = nd.relu(self.conv(x))
        #         print(conv_result.shape)

        # 使用mask 将卷积结果分段
        be1_mask = nd.zeros(conv_result.shape, ctx=CTX)
        aes_mask = nd.zeros(conv_result.shape, ctx=CTX)
        be2_mask = nd.zeros(conv_result.shape, ctx=CTX)
        #
        be1_pad = nd.ones(conv_result.shape, ctx=CTX) * (-100)
        aes_pad = nd.ones(conv_result.shape, ctx=CTX) * (-100)
        be2_pad = nd.ones(conv_result.shape, ctx=CTX) * (-100)
        #         print(be1_mask.shape)
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

        # 卷积结果*mask得到分段后的卷积向量
        be1 = conv_result * be1_mask
        aes = conv_result * aes_mask
        be2 = conv_result * be2_mask
        # 将无用部分赋予最小值，避免影响maxpool操作
        be1 = be1 + be1_pad
        aes = aes + aes_pad
        be2 = be2 + be2_pad

        o1 = self.pmp(be1)
        o2 = self.pmp(aes)
        o3 = self.pmp(be2)

        out = nd.concat(o1, o2, o3, dim=2)
        y = self.output(out)
        return y


net = PCNN()
net.collect_params().initialize(init=init.Xavier(), ctx=ctx)

print(net)

batch_size = 100
num_epochs = 100
decay_rate = 0.1
gap = 25
loss = gloss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'AdaDelta', {'rho': 0.95, 'epsilon': 1e-6, 'wd': 0.01})
# trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

if ADAPTIVE_LEARNING_RATE:
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'beta1': 0.9, 'beta2': 0.99, 'learning_rate': 1e-2})
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
        train_loss_sum = 0
        train_acc_sum = 0
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0 and trainer.learning_rate > 0.0001:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
            print("learning_rate decay: %f" % trainer.learning_rate)
        start = time.time()
        for X, y in train_iter:
            y = y.copyto(CTX)
            with autograd.record():
                y_hat = net(X)
                lss = loss(y_hat, y)
            lss.backward()
            trainer.step(batch_size)
            train_loss_sum += lss.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
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
    result = nd.concat(y_test.expand_dims(axis=1), y_hat, dim=1)
    np.save("result_pcnn_SemEval.npy", result.asnumpy())
    predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
    label_list = y_test.astype(np.int).asnumpy().tolist()
    print(precision_recall_fscore_support(label_list, predict_list, average='macro'))
    print(classification_report(label_list, predict_list))


train(net, train_data, test_data, loss, num_epochs, batch_size, trainer)
