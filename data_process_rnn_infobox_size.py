# 添加可INFOBOX实际尺寸，用于可变卷积核

from gensim.models import KeyedVectors
import numpy as np
import os
import pickle

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train2.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test2.txt")
DIMENSION = 100
INFOBOX_VALUE_LENGTH = 10
INFOBOX_LENGTH = 20
FIXED_WORD_LENGTH = 60
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
PLACEHOLDER = np.zeros(DIMENSION)

infobox_key = []
infobox_value = np.load('infobox2vec_size_value.npy')
with open("infobox2vec_size_key.txt", "r", encoding="utf8") as f:
    for line in f:
        infobox_key.append(line.strip())
with open("infobox2vec_size.txt", "rb") as f:
    infobox_size_dict = pickle.load(f)


def get_entity_infobox(entity_name):
    if entity_name in infobox_key:
        idx = infobox_key.index(entity_name)
        return infobox_value[idx]
    else:
        return np.zeros(infobox_value[0].shape)


def get_entity_size(entity_name):
    if entity_name in infobox_size_dict:
        return infobox_size_dict[entity_name].copy()
    else:
        return []


for corpus, save_filename in ((CORPUS_TRAIN, "data_train_rnn_infobox_size.npy"),
                              (CORPUS_TEST, "data_test_rnn_infobox_size.npy")):
    output_sentence = []
    output_relation = []
    output_en1_infobox = []
    output_en2_infobox = []
    output_en1_kernel_num = []
    output_en2_kernel_num = []
    output_en1_size = []
    output_en2_size = []
    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split()
            entity_a = content[0]
            entity_b = content[1]
            relation = int(content[2])
            sentence = content[3:]

            sentence_vector = []

            for i in range(len(sentence)):
                if sentence[i] == entity_a:
                    entity_a_pos = i
                if sentence[i] == entity_b:
                    entity_b_pos = i

                if sentence[i] not in wordvec:
                    word_vector = PLACEHOLDER
                else:
                    word_vector = wordvec[sentence[i]]
                sentence_vector.append(word_vector)

            if len(sentence_vector) < FIXED_WORD_LENGTH:
                for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                    sentence_vector.append(PLACEHOLDER)

            output_sentence.append(sentence_vector)
            output_relation.append(relation)

            en1_vec = get_entity_infobox(entity_a)
            output_en1_infobox.append(en1_vec)
            sz1 = get_entity_size(entity_a)
            output_en1_kernel_num.append(len(sz1))
            if len(sz1) < INFOBOX_LENGTH:
                for ii in range(INFOBOX_LENGTH - len(sz1)):
                    sz1.append(0)
            output_en1_size.append(sz1)

            en2_vec = get_entity_infobox(entity_b)
            output_en2_infobox.append(en2_vec)
            sz2 = get_entity_size(entity_b)
            output_en2_kernel_num.append(len(sz2))
            if len(sz2) < INFOBOX_LENGTH:
                for ii in range(INFOBOX_LENGTH - len(sz2)):
                    sz2.append(0)
            output_en2_size.append(sz2)

    print("length of output_sentence: %d" % len(output_sentence))

    np_sentence = np.array(output_sentence, dtype=float)
    np_relation = np.array(output_relation, dtype=int)
    np_en1_infobox = np.array(output_en1_infobox, dtype=float)
    np_en2_infobox = np.array(output_en2_infobox, dtype=float)
    np_en1_size = np.array(output_en1_size, dtype=int)
    np_en2_size = np.array(output_en2_size, dtype=int)
    np_en1_kernel_num = np.array(output_en1_kernel_num, dtype=int)
    np_en2_kernel_num = np.array(output_en2_kernel_num, dtype=int)

    print(np_sentence.shape)
    print(np_en1_infobox.shape)
    print(np_en2_infobox.shape)
    print(np_en1_kernel_num.shape)
    print(np_en1_size.shape)
    assert np_en1_size.shape[1] == INFOBOX_LENGTH
    assert np_en2_size.shape[1] == INFOBOX_LENGTH

    sentence_vec = np_sentence.reshape(np_sentence.shape[0],
                                       DIMENSION * FIXED_WORD_LENGTH)
    np_en_size = np.concatenate((np_en1_size, np_en2_size), axis=1)
    np_en_infobox = np.concatenate((np_en1_infobox.reshape(np_en1_infobox.shape[0], -1),
                                    np_en2_infobox.reshape(np_en2_infobox.shape[0], -1)), axis=1)
    # relation + sentence_vec
    conc = np.concatenate((np.expand_dims(np_relation, axis=1),
                           sentence_vec,
                           np.expand_dims(np_en1_kernel_num, axis=1),
                           np.expand_dims(np_en2_kernel_num, axis=1),
                           np_en_size,
                           np_en_infobox),
                          axis=1)
    print(conc.shape)

    tag_0 = conc[conc[:, 0] == 0]
    tag_1 = conc[conc[:, 0] == 1]
    tag_2 = conc[conc[:, 0] == 2]
    tag_3 = conc[conc[:, 0] == 3]
    tag_4 = conc[conc[:, 0] == 4]
    tag_5 = conc[conc[:, 0] == 5]
    tag_6 = conc[conc[:, 0] == 6]

    filter = np.concatenate((
        tag_0, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6), axis=0)
    print(filter.shape)

    np.random.shuffle(filter)
    np.save(save_filename, filter)
