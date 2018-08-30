import mxnet as mx
from gensim.models import KeyedVectors
import numpy as np
import os

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "\\wordvectors.kv")
CORPUS = os.path.join(CWD, "\\separated_corpus_with_label_patch_amend.txt")
DIMENSION = 100
POS_DIMENSION = 5
FIXED_WORD_LENGTH = 60
TRAIN_RADIO = 0.7

wordvec = KeyedVectors.load(WORDVEC, mmap='r')
wordvec['UNK'] = np.zeros(DIMENSION)
wordvec['BLANK'] = np.zeros(DIMENSION)

POS_VECTOR = np.random.random((FIXED_WORD_LENGTH * 2, POS_DIMENSION))

output_entity_pos = []
output_relative_pos = []
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
        entity_pos = []
        relative_pos = []
        entity_a_pos_list = []  # 取实体a与实体b最接近的位置
        entity_b_pos_list = []
        entity_a_pos = -1
        entity_b_pos = -1
        for i in range(len(sentence)):
            if sentence[i] == entity_a:
                entity_a_pos_list.append(i)
                # entity_a_pos = i
            if sentence[i] == entity_b:
                entity_b_pos_list.append(i)
                # entity_b_pos = i

            if sentence[i] not in wordvec:
                word_vector = wordvec['UNK']
            else:
                word_vector = wordvec[sentence[i]]
            sentence_vector.append(word_vector)

        d_pos = FIXED_WORD_LENGTH
        for i in entity_a_pos_list:
            for j in entity_b_pos_list:
                if abs(i - j) < d_pos:
                    d_pos = abs(i - j)
                    entity_a_pos = i
                    entity_b_pos = j
        exception_flag = False
        if entity_a_pos == -1 or entity_b_pos == -1:
            print(
                "entity not found: (%s, %d) (%s, %d) @%s" % (entity_a, entity_a_pos, entity_b, entity_b_pos, sentence))
            exception_flag = True
        if entity_a_pos < entity_b_pos:
            entity_pos.append([entity_a_pos, entity_b_pos])
        elif entity_a_pos > entity_b_pos:
            entity_pos.append([entity_b_pos, entity_a_pos])
        else:
            print("entity equal: (%s, %d) (%s, %d) @%s" % (entity_a, entity_a_pos, entity_b, entity_b_pos, sentence))
            exception_flag = True
            # exit(1)
        if exception_flag:
            if relation == -1:
                continue
            print(line)
            exit(1)
        for i in range(len(sentence)):
            relative_vector_entity_a = POS_VECTOR[i - entity_a_pos, :]
            relative_vector_entity_b = POS_VECTOR[i - entity_b_pos, :]
            pos_vec = np.concatenate((relative_vector_entity_a, relative_vector_entity_b))
            relative_pos.append(pos_vec)
        if len(sentence_vector) < FIXED_WORD_LENGTH:
            for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                sentence_vector.append(wordvec['BLANK'])
                pos_vec = np.concatenate((POS_VECTOR[FIXED_WORD_LENGTH, :], POS_VECTOR[FIXED_WORD_LENGTH, :]))
                relative_pos.append(pos_vec)

        output_sentence.append(sentence_vector)
        output_relation.append(relation)
        output_entity_pos.append(entity_pos)
        output_relative_pos.append(relative_pos)

print("length of output_sentence: %d" % len(output_sentence))

np_sentence = np.array(output_sentence, dtype=float)
np_relation = np.array(output_relation, dtype=int)
np_entity_pos = np.array(output_entity_pos, dtype=int)
np_relative_pos = np.array(output_relative_pos, dtype=float)
print(np_sentence.shape)
print(np_relative_pos.shape)
print(np_entity_pos.shape)

np_sentence_matrix = np.concatenate((np_sentence, np_relative_pos), axis=2)
print(np_sentence_matrix.shape)
sentence_vec = np_sentence_matrix.reshape(np_sentence_matrix.shape[0],
                                          (DIMENSION + 2 * POS_DIMENSION) * FIXED_WORD_LENGTH)
entity_pos_vec = np_entity_pos.reshape(np_entity_pos.shape[0], 2)

# relation + entity position + sentence_vec
conc = np.concatenate((np.expand_dims(np_relation, axis=1), entity_pos_vec, sentence_vec), axis=1)
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

# tag_0[:, 0] = 0
# tag_8[:, 0] = 4
# tag_9[:, 0] = 8
# tag_10[:, 0] = 9
tag_0[:, 0] = 0
# tag_2[:, 0] = 1
# tag_6[:, 0] = 1
# tag_10[:, 0] = 1
# tag_3[:, 0] = 2
# tag_4[:, 0] = 3
# tag_8[:, 0] = 3
# tag_5[:, 0] = 4
# tag_7[:, 0] = 5
# tag_9[:, 0] = 6

tag_0 = tag_0[:3753, :]

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
np.save('data_train.npy', filter_train)
np.save('data_test.npy', filter_test)
