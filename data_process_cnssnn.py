import numpy as np

SENTENCE_DIMENSION = 100
POS_DIMENSION = 5
DIMENSION = SENTENCE_DIMENSION + 2 * POS_DIMENSION
FIXED_WORD_LENGTH = 60

input_train = np.load('data_train.npy')
input_test = np.load('data_test.npy')

entityvec_key = []
entityvec_value = np.load('entity2vec_value.npy')

with open("entity2vec_key.txt","r",encoding="utf8") as f:
    for line in f:
        entityvec_key.append(line.strip())
