import mxnet as mx
from gensim.models import KeyedVectors
import numpy as np

WORDVEC = "D:\\Projects\\Baike\\wordvectors.kv"
CORPUS = "D:\\Projects\\Baike\\separated_corpus_with_label.txt"
DIM = 100
FIXED_WORD_LENGTH = 78
TRAIN_RADIO = 0.7

wordvec = KeyedVectors.load(WORDVEC, mmap='r')
wordvec['UNK'] = np.zeros(DIM)
wordvec['BLANK'] = np.zeros(DIM)

output_sentence = []
output_relation = []
with open(CORPUS, "r", encoding="utf8") as f:
    for line in f:
        content = line.strip().split()
        sentence = content[:-1]
        relation = content[-1]
        sentence_vector = []
        for w in sentence:
            if w not in wordvec:
                word_vector = wordvec['UNK']
            else:
                word_vector = wordvec[w]
            sentence_vector.append(word_vector)
        if len(sentence_vector) < FIXED_WORD_LENGTH:
            for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                sentence_vector.append(wordvec['BLANK'])

        output_sentence.append(sentence_vector)
        output_relation.append(relation)

print("length of output_sentence: %d" % len(output_sentence))

np_sentence = np.array(output_sentence, dtype=float)
np_relation = np.array(output_relation, dtype=int)
np_sentence_matrix = np_sentence.reshape(np_sentence.shape[0], FIXED_WORD_LENGTH * DIM)
np_sentence_matrix = np.concatenate((np_sentence_matrix, np.expand_dims(np_relation, axis=1)), 1)

tag_1 = np_sentence_matrix[np_sentence_matrix[:, -1] == 1]
tag_2 = np_sentence_matrix[np_sentence_matrix[:, -1] == 2]
tag_3 = np_sentence_matrix[np_sentence_matrix[:, -1] == 3]
tag_4 = np_sentence_matrix[np_sentence_matrix[:, -1] == 4]
tag_5 = np_sentence_matrix[np_sentence_matrix[:, -1] == 5]
tag_6 = np_sentence_matrix[np_sentence_matrix[:, -1] == 6]
tag_7 = np_sentence_matrix[np_sentence_matrix[:, -1] == 7]
tag_8 = np_sentence_matrix[np_sentence_matrix[:, -1] == 8]
tag_9 = np_sentence_matrix[np_sentence_matrix[:, -1] == 9]
tag_10 = np_sentence_matrix[np_sentence_matrix[:, -1] == 10]

tag_1_train = tag_1[:int(TRAIN_RADIO * len(tag_1))]
tag_1_test = tag_1[int(TRAIN_RADIO * len(tag_1)):]
tag_2_train = tag_2[:int(TRAIN_RADIO * len(tag_2))]
tag_2_test = tag_2[int(TRAIN_RADIO * len(tag_2)):]
tag_3_train = tag_3[:int(TRAIN_RADIO * len(tag_3))]
tag_3_test = tag_3[int(TRAIN_RADIO * len(tag_3)):]
tag_4_train = tag_4[:int(TRAIN_RADIO * len(tag_4))]
tag_4_test = tag_4[int(TRAIN_RADIO * len(tag_4)):]
tag_5_train = tag_5[:int(TRAIN_RADIO * len(tag_5))]
tag_5_test = tag_5[int(TRAIN_RADIO * len(tag_5)):]
tag_6_train = tag_6[:int(TRAIN_RADIO * len(tag_6))]
tag_6_test = tag_6[int(TRAIN_RADIO * len(tag_6)):]
tag_7_train = tag_7[:int(TRAIN_RADIO * len(tag_7))]
tag_7_test = tag_7[int(TRAIN_RADIO * len(tag_7)):]
tag_8_train = tag_8[:int(TRAIN_RADIO * len(tag_8))]
tag_8_test = tag_8[int(TRAIN_RADIO * len(tag_8)):]
tag_9_train = tag_9[:int(TRAIN_RADIO * len(tag_9))]
tag_9_test = tag_9[int(TRAIN_RADIO * len(tag_9)):]
tag_10_train = tag_10[:int(TRAIN_RADIO * len(tag_10))]
tag_10_test = tag_10[int(TRAIN_RADIO * len(tag_10)):]

filter_train = np.concatenate((
    tag_1_train, tag_2_train, tag_3_train, tag_4_train, tag_5_train, tag_6_train, tag_7_train,
    tag_8_train, tag_9_train, tag_10_train), axis=0)
filter_test = np.concatenate((
    tag_1_test, tag_2_test, tag_3_test, tag_4_test, tag_5_test, tag_6_test, tag_7_test,
    tag_8_test, tag_9_test, tag_10_test), axis=0)
print(filter_train.shape)
print(filter_test.shape)

np.random.shuffle(filter_train)
np.random.shuffle(filter_test)
np.save('data_train.npy', filter_train)
np.save('data_test.npy', filter_test)
