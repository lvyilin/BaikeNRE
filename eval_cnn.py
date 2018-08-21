import os

import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from sklearn.metrics import precision_recall_fscore_support

CWD = os.getcwd()
MODEL_PARAMS_PATH = CWD + "\\net_params\\cnn\\net_cnn_epoch80.params"
SENTENCE_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = SENTENCE_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60

input_train = np.load('data_train.npy')
input_test = np.load('data_test.npy')
x_train = input_train[:, 3:].reshape((input_train.shape[0], FIXED_WORD_LENGTH, DIMENSION))
x_train = np.expand_dims(x_train, axis=1)
y_train = input_train[:, 0]
print(x_train.shape)
print(y_train.shape)
x_test = input_test[:, 3:].reshape((input_test.shape[0], FIXED_WORD_LENGTH, DIMENSION))
x_test = np.expand_dims(x_test, axis=1)
y_test = input_test[:, 0]
print(x_test.shape)
print(y_test.shape)

x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)
x_all = x_all.astype(np.float32)
y_all = y_all.astype(np.int)
print(x_all.shape, y_all.shape)

net = nn.Sequential()
# CNN, CR-CNN
# net.add(nn.Conv2D(256, kernel_size=(5, DIMENSION), padding=(1, 0), activation='relu'))
net.add(nn.Conv2D(256, kernel_size=(3, DIMENSION), padding=(1, 0), activation='relu'))
# net.add(nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH - 2, 1)))
net.add(nn.MaxPool2D(pool_size=(FIXED_WORD_LENGTH, 1)))
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dropout(0.5))
net.add(nn.Dense(6))

net.load_parameters(MODEL_PARAMS_PATH)
print(net)

label_list = y_all.tolist()
y_hat = net(nd.array(x_all))
predict_list = y_hat.argmax(axis=1).asnumpy().astype(np.int).tolist()
print(precision_recall_fscore_support(label_list, predict_list, average='macro'))
