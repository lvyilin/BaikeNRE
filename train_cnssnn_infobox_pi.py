import os
import time

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd, init
from mxnet.base import numeric_types
from mxnet.gluon import loss as gloss, nn, rnn
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn.conv_layers import _infer_weight_shape
from sklearn.metrics import precision_recall_fscore_support, classification_report

CWD = os.getcwd()
SAVE_MODEL_PATH = os.path.join(CWD, "net_params", "cnssnn_infobox_pi", "net_cnssnn_infobox_pi_epoch%d.params")
WORD_DIMENSION = 100
DIMENSION = WORD_DIMENSION
FIXED_WORD_LENGTH = 60
INFOBOX_VALUE_LENGTH = 10
INFOBOX_LENGTH = 20
ADAPTIVE_LEARNING_RATE = True
MAX_ENTITY_DEGREE = 50
ENTITY_DEGREE = MAX_ENTITY_DEGREE + 1
MASK_LENGTH = ENTITY_DEGREE
ENTITY_EDGE_VEC_LENGTH = ENTITY_DEGREE * (WORD_DIMENSION * 2)
VEC_LENGTH = DIMENSION * FIXED_WORD_LENGTH + ENTITY_EDGE_VEC_LENGTH * 2

CTX = mx.gpu(1)
ctx = [CTX]

input_train = np.load('data_train_cnssnn_infobox_pi.npy')
input_test = np.load('data_test_cnssnn_infobox_pi.npy')

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
            self.dense1 = nn.Dense(68, activation="sigmoid")
            self.dense2 = nn.Dense(68, activation="sigmoid")
            # cnssnn
            self.center_att = nn.Sequential()
            self.center_att.add(nn.Dense(1, in_units=200, flatten=False,
                                         activation="sigmoid"))
            self.center_out = nn.Sequential()
            self.center_out.add(nn.Dense(100, activation="relu"))
            # --------
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
        e1_vec_start = e2_start + INFOBOX_LENGTH * INFOBOX_VALUE_LENGTH * DIMENSION
        e1neimask = input_data[:, e1_vec_start:e1_vec_start + MASK_LENGTH]  # (m, 51)
        e1edge = input_data[:, e1_vec_start + MASK_LENGTH:e1_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH].reshape(
            (input_data.shape[0], ENTITY_DEGREE, WORD_DIMENSION * 2))  # (m, 51, 200)
        e1neigh = e1edge[:, :, :WORD_DIMENSION]

        e2_vec_start = e1_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH
        e2neimask = input_data[:, e2_vec_start:e2_vec_start + MASK_LENGTH]  # (m, 51)
        e2edge = input_data[:, e2_vec_start + MASK_LENGTH:e2_vec_start + MASK_LENGTH + ENTITY_EDGE_VEC_LENGTH].reshape(
            (input_data.shape[0], ENTITY_DEGREE, WORD_DIMENSION * 2))  # (m, 51,200)
        e2neigh = e2edge[:, :, :WORD_DIMENSION]
        # SENTENCE
        h_sen = self.lstm(x_sen)  # (batch_size,60,128)

        # CNSSNN
        att = self.center_att
        e1edge = nd.tanh(e1edge)
        e1g = att(e1edge)  # (m,51,1)
        #         print(e1g)
        e1g = e1g * e1neimask.expand_dims(2)
        e1g = nd.softmax(e1g, axis=1)
        e1gt = nd.transpose(e1g, axes=(0, 2, 1))  # (m,1,151)
        e1n = nd.batch_dot(e1gt, e1neigh)  # (m,1,100)
        e1n = e1n.reshape((e1n.shape[0], 100))  # (m,100)

        e2edge = nd.tanh(e2edge)
        e2g = att(e2edge)  # (m,51,1)
        #         print(e2g)
        e2g = e2g * e2neimask.expand_dims(2)
        e2g = nd.softmax(e2g, axis=1)
        e2gt = nd.transpose(e2g, axes=(0, 2, 1))  # (m,1,151)
        e2n = nd.batch_dot(e2gt, e2neigh)  # (m,1,100)
        e2n = e2n.reshape((e2n.shape[0], 100))  # (m,100)

        y_edge = nd.concat(e1n, e2n, dim=1)  # (m,200)
        center_out = self.center_out
        y_edge = center_out(y_edge)

        # INFOBOX
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
        h_sen_infobox = nd.concat(h_sen_new, y_edge, e_infobox_list_all_att, dim=1)
        y = self.output(h_sen_infobox)
        return y


net = Network()
# net.load_params(SAVE_MODEL_PATH % 3, ctx=CTX)
net.collect_params().initialize(init=init.Xavier(), ctx=ctx)
print(net)

batch_size = 128
num_epochs = 100
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
        train_l_sum = 0
        train_acc_sum = 0
        if ADAPTIVE_LEARNING_RATE and epoch % gap == 0:
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
    np.save("cnssnn_infobox_pi_epoch.npy", result.asnumpy())
    predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
    label_list = y_test.astype(np.int).asnumpy().tolist()
    print(precision_recall_fscore_support(label_list, predict_list, average='weighted'))
    print(classification_report(label_list, predict_list))


train(net, train_data, test_data, loss, num_epochs, batch_size, trainer)
