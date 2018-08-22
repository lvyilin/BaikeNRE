import os

import numpy as np
from mxnet import nd
from mxnet.gluon import nn, rnn
from sklearn.metrics import precision_recall_fscore_support

CWD = os.getcwd()
MODEL_PARAMS_PATH = CWD + "\\net_params\\gru_wpe\\net_gru_wpe_epoch80.params"
SENTENCE_DIMENSION = 100
DIMENSION = 110
FIXED_WORD_LENGTH = 60

input_test = np.load('data_test.npy')
x_test = input_test[:, 3:].reshape((input_test.shape[0], FIXED_WORD_LENGTH, DIMENSION))
# x_test = np.expand_dims(x_test, axis=1)
y_test = input_test[:, 0]
print(x_test.shape)
print(y_test.shape)

x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

x_all = x_test
y_all = y_test
x_all = x_all.astype(np.float32)
y_all = y_all.astype(np.int)
print(x_all.shape, y_all.shape)


class Network(nn.Block):
    def __init__(self, prefix=None, params=None):
        super().__init__(prefix, params)

        self.gru = rnn.GRU(64, num_layers=1, bidirectional=True, dropout=0.2)
        self.output = nn.Dense(6)

    def forward(self, input_data):
        x = nd.transpose(input_data, axes=(1, 0, 2))
        h = nd.transpose(self.gru(x), axes=(1, 0, 2))
        return self.output(h)


net = Network()
net.load_parameters(MODEL_PARAMS_PATH)
print(net)

label_list = y_all.tolist()
y_hat = net(nd.array(x_all))
predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
print(precision_recall_fscore_support(label_list, predict_list, average='macro'))
