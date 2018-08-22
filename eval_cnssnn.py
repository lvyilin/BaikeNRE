import os

import numpy as np
from mxnet import nd
from mxnet.gluon import nn, rnn
from sklearn.metrics import precision_recall_fscore_support

CWD = os.getcwd()
MODEL_PARAMS_PATH = CWD + "\\net_params\\cnssnn\\net_cnssnn_epoch80.params"
WORD_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = WORD_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60
MAX_ENTITY_DEGREE = 50
ENTITY_DEGREE = MAX_ENTITY_DEGREE + 1
MASK_LENGTH = ENTITY_DEGREE
ENTITY_EDGE_VEC_LENGTH = ENTITY_DEGREE * (WORD_DIMENSION * 2)
VEC_LENGTH = DIMENSION * FIXED_WORD_LENGTH + ENTITY_EDGE_VEC_LENGTH * 2

input_test = np.load('data_test_cnssnn.npy')


x_test = input_test[:, 3:]
y_test = input_test[:, 0]
print(x_test.shape)
print(y_test.shape)

x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

x_all = x_test
y_all = y_test
x_all = x_all.astype(np.float32)
y_all = y_all.astype(np.int)
print(x_all.shape, y_all.shape)


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
            self.output.add(nn.Dense(6))

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
net.load_parameters(MODEL_PARAMS_PATH)
print(net)

label_list = y_all.tolist()
y_hat = net(nd.array(x_all))
predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
print(precision_recall_fscore_support(label_list, predict_list, average='macro'))
