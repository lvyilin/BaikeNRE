import time
import os
import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss, nn, rnn
from mxnet.base import numeric_types
from mxnet.gluon.nn.conv_layers import _infer_weight_shape
from mxnet.gluon.nn import Activation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold

MARK = "rnn_infobox_pi"
CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", MARK, "net_{}_epoch%d.params".format(MARK))
WORD_DIMENSION = 100
DIMENSION = WORD_DIMENSION
FIXED_WORD_LENGTH = 60
INFOBOX_VALUE_LENGTH = 10
INFOBOX_LENGTH = 20
ADAPTIVE_LEARNING_RATE = True

CTX = mx.gpu(3)
ctx = [CTX]

input_train = np.load('data_train_rnn_infobox_pi.npy'.format(MARK))
input_test = np.load('data_test_rnn_infobox_pi.npy'.format(MARK))
input_all = np.concatenate((input_train, input_test), axis=0)
np.random.shuffle(input_all)
input_x = input_all[:, 3:].astype(np.float32)
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


class _Conv(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides, padding, dilation,
                 groups, layout, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 op_name='Convolution', adj=None, prefix=None, params=None):
        super(_Conv, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(strides, numeric_types):
                strides = (strides,) * len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,) * len(kernel_size)
            if isinstance(dilation, numeric_types):
                dilation = (dilation,) * len(kernel_size)
            self._op_name = op_name
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}
            if adj is not None:
                self._kwargs['adj'] = adj

            dshape = [0] * (len(kernel_size) + 2)
            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_channels
            wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = Activation(activation, prefix=activation + '_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, kernel, weight, bias=None):
        weight = kernel
        if bias is None:
            act = getattr(F, self._op_name)(x, weight, name='fwd', **self._kwargs)
        else:
            act = getattr(F, self._op_name)(x, weight, bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)


class MyConv2D(_Conv):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,) * 2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(MyConv2D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class Network(nn.Block):
    def __init__(self, prefix=None, params=None):
        super().__init__(prefix, params)
        with self.name_scope():
            self.lstm = rnn.LSTM(64, num_layers=1, bidirectional=True, dropout=0.2, layout='NTC')
            self.lstm_out = nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1))
            #             self.att = nn.Sequential()
            #             self.att.add(nn.Dense(1, flatten=False,
            #                                   activation="tanh"))
            self.conv1 = MyConv2D(INFOBOX_LENGTH, kernel_size=(INFOBOX_VALUE_LENGTH, DIMENSION),
                                  strides=(1, 1), dilation=(1, 1), use_bias=False, in_channels=1,
                                  activation='relu')
            self.conv2 = MyConv2D(INFOBOX_LENGTH, kernel_size=(INFOBOX_VALUE_LENGTH, DIMENSION),
                                  strides=(1, 1), dilation=(1, 1), use_bias=False, in_channels=1,
                                  activation='relu')
            #             self.pool = nn.MaxPool2D(pool_size=(10,1), strides=(1, 1))
            self.dense1 = nn.Dense(384, activation="sigmoid")
            self.dense2 = nn.Dense(384, activation="sigmoid")
            self.output = nn.Sequential()
            self.output.add(nn.Flatten())
            self.output.add(nn.Activation(activation='relu'))
            self.output.add(nn.Dropout(0.5))
            self.output.add(nn.Dense(7))

    def forward(self, input_data):
        x_sen = input_data[:, :DIMENSION * FIXED_WORD_LENGTH].reshape(
            (input_data.shape[0], FIXED_WORD_LENGTH, DIMENSION))
        e1_start = DIMENSION * FIXED_WORD_LENGTH
        e1_infobox = input_data[:, e1_start:e1_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION].reshape(
            (input_data.shape[0], INFOBOX_LENGTH, INFOBOX_VALUE_LENGTH,
             DIMENSION))  # (batch_size,INFOBOX_LENGTH,INFOBOX_VALUE_LENGTH,100)
        e2_start = e1_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION
        e2_infobox = input_data[:, e2_start:e2_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION].reshape(
            (input_data.shape[0], INFOBOX_LENGTH, INFOBOX_VALUE_LENGTH,
             DIMENSION))  # (batch_size,INFOBOX_LENGTH,INFOBOX_VALUE_LENGTH,100)
        h_sen = self.lstm(x_sen)  # (batch_size,60,128)

        e1_infobox_list_all = nd.ones((e1_infobox.shape[0], e1_infobox.shape[1], 51, 1),
                                      ctx=CTX)  # (batch_size,INFOBOX_LENGTH,51,1)
        e2_infobox_list_all = nd.ones((e1_infobox.shape[0], e2_infobox.shape[1], 51, 1),
                                      ctx=CTX)  # (batch_size,INFOBOX_LENGTH,51,1)

        for i in range(e1_infobox.shape[0]):
            e1 = self.conv1(x_sen[i].expand_dims(axis=0).expand_dims(axis=1), e1_infobox[i].expand_dims(axis=1))
            #             e1_p = self.pool(e1)
            e1_infobox_list_all[i] = e1.reshape((e1.shape[1], e1.shape[2], e1.shape[3]))
            e2 = self.conv2(x_sen[i].expand_dims(axis=0).expand_dims(axis=1), e2_infobox[i].expand_dims(axis=1))
            #             e2_p = self.pool(e2)
            e2_infobox_list_all[i] = e2.reshape((e2.shape[1], e2.shape[2], e2.shape[3]))

        e1_infobox_list_all = e1_infobox_list_all.reshape(
            (e1_infobox.shape[0], e1_infobox.shape[1], -1))  # (batch_size,INFOBOX_LENGTH,51)
        e2_infobox_list_all = e2_infobox_list_all.reshape(
            (e2_infobox.shape[0], e2_infobox.shape[1], -1))  # (batch_size,INFOBOX_LENGTH,51)

        e1_infobox_list_all_new = self.dense1(e1_infobox_list_all)
        e2_infobox_list_all_new = self.dense2(e2_infobox_list_all)

        #         g1 = nd.softmax(self.att(e1_infobox_list_all),axis=2) # (batch_size,INFOBOX_LENGTH,1)
        #         g2 = nd.softmax(self.att(e2_infobox_list_all),axis=2) # (batch_size,INFOBOX_LENGTH,1)
        #         g1_att = nd.batch_dot(nd.transpose(g1,axes = (0,2,1)),e1_infobox_list_all) # (batch_size,1,64)
        #         g2_att = nd.batch_dot(nd.transpose(g2,axes = (0,2,1)),e2_infobox_list_all) # (batch_size,1,64)
        #         g1_att = g1_att.reshape((g1_att.shape[0],-1)) # (batch_size,64)
        #         g2_att = g2_att.reshape((g2_att.shape[0],-1)) # (batch_size,64)

        # (batch_size,128)
        e_infobox_list_all_att = nd.concat(e1_infobox_list_all_new, e2_infobox_list_all_new, dim=1)
        h_sen_new = self.lstm_out(h_sen.expand_dims(1))
        h_sen_new = h_sen_new.reshape((h_sen_new.shape[0], -1))  # (batch_size,128)
        # (batch_size,256)
        h_sen_infobox = nd.concat(h_sen_new, e_infobox_list_all_att, dim=1)
        y = self.output(h_sen_infobox)
        return y


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
            trainer.step(batch_size, ignore_stale_grad=True)
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
num_epochs = 50
decay_rate = 0.1
lowest_lr = 0.0001
gap = 20
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
