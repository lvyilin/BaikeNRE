import mxnet as mx
from gensim.models import KeyedVectors
import numpy as np
import os

CWD = os.getcwd()
WORDVEC = CWD + "\\wordvectors.kv"
CORPUS = CWD + "\\separated_corpus_with_label_patch_amend.txt"
DIMENSION = 100
FIXED_WORD_LENGTH = 60
TRAIN_RADIO = 0.7

wordvec = KeyedVectors.load(WORDVEC, mmap='r')
wordvec['UNK'] = np.zeros(DIMENSION)
wordvec['BLANK'] = np.zeros(DIMENSION)

output_sentence = []
output_relation = []
with open(CORPUS, "r", encoding="utf8") as f:
    for line in f:
        content = line.strip().split()
        entity_a = content[0]
        entity_b = content[1]
        try:
            relation = int(content[2])
        except ValueError:
            continue
        sentence = content[3:]

        sentence_vector = []

        for i in range(len(sentence)):
            if sentence[i] == entity_a:
                entity_a_pos = i
            if sentence[i] == entity_b:
                entity_b_pos = i

            if sentence[i] not in wordvec:
                word_vector = wordvec['UNK']
            else:
                word_vector = wordvec[sentence[i]]
            sentence_vector.append(word_vector)

        if len(sentence_vector) < FIXED_WORD_LENGTH:
            for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                sentence_vector.append(wordvec['BLANK'])

        output_sentence.append(sentence_vector)
        output_relation.append(relation)

print("length of output_sentence: %d" % len(output_sentence))

np_sentence = np.array(output_sentence, dtype=float)
np_relation = np.array(output_relation, dtype=int)

print(np_sentence.shape)

sentence_vec = np_sentence.reshape(np_sentence.shape[0],
                                   DIMENSION * FIXED_WORD_LENGTH)

# relation + sentence_vec
conc = np.concatenate((np.expand_dims(np_relation, axis=1), sentence_vec), axis=1)
print(conc.shape)

tag_1 = conc[conc[:, 0] == 1]
tag_2 = conc[conc[:, 0] == 2]
tag_3 = conc[conc[:, 0] == 3]
tag_4 = conc[conc[:, 0] == 4]
tag_5 = conc[conc[:, 0] == 5]
tag_6 = conc[conc[:, 0] == 6]
tag_7 = conc[conc[:, 0] == 7]
tag_8 = conc[conc[:, 0] == 8]
tag_9 = conc[conc[:, 0] == 9]
tag_10 = conc[conc[:, 0] == 10]
tag_0 = conc[conc[:, 0] == -1]
tag_0[:, 0] = 0

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
tag_0_train = tag_0[:int(TRAIN_RADIO * len(tag_0))]
tag_0_test = tag_0[int(TRAIN_RADIO * len(tag_0)):]
# filter_train = np.concatenate((
#     tag_1_train, tag_3_train, tag_4_train, tag_6_train, tag_7_train,
#     tag_9_train), axis=0)
# filter_test = np.concatenate((
#     tag_1_test, tag_3_test, tag_4_test, tag_6_test, tag_7_test,
#     tag_9_test), axis=0)

filter_train = np.concatenate((
    tag_1_train, tag_2_train, tag_3_train, tag_4_train, tag_5_train, tag_6_train, tag_7_train,
    tag_8_train, tag_9_train, tag_10_train, tag_0_train), axis=0)
filter_test = np.concatenate((
    tag_1_test, tag_2_test, tag_3_test, tag_4_test, tag_5_test, tag_6_test, tag_7_test,
    tag_8_test, tag_9_test, tag_10_test, tag_0_test), axis=0)
print(filter_train.shape)
print(filter_test.shape)

np.random.shuffle(filter_train)
np.random.shuffle(filter_test)
np.save('data_train_rnn.npy', filter_train)
np.save('data_test_rnn.npy', filter_test)
