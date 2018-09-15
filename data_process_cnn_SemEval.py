import mxnet as mx
from gensim.models import KeyedVectors
import numpy as np
import os

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train_SemEval.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test_SemEval.txt")
DIMENSION = 100
POS_DIMENSION = 5
FIXED_WORD_LENGTH = 60

wordvec = KeyedVectors.load(WORDVEC, mmap='r')
PLACEHOLDER = np.zeros(DIMENSION)
POS_VECTOR = np.random.random((FIXED_WORD_LENGTH * 2, POS_DIMENSION))

for corpus, save_filename in ((CORPUS_TRAIN, "data_train_cnn_SemEval.npy"),
                              (CORPUS_TEST, "data_test_cnn_SemEval.npy")):
    output_entity_pos = []
    output_relative_pos = []
    output_sentence = []
    output_relation = []
    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split("\t")
            idx = int(content[0])
            en1 = content[1]
            en2 = content[2]
            en1_pos = int(content[3])
            en2_pos = int(content[4])
            relation = int(content[5])
            sentence = content[6].split(" ")
            if len(sentence) > FIXED_WORD_LENGTH:
                sentence[:] = sentence[:FIXED_WORD_LENGTH]
            sentence_vector = []
            relative_pos = []
            for i in range(len(sentence)):
                word_vector = wordvec[sentence[i]] if sentence[i] in wordvec else PLACEHOLDER
                sentence_vector.append(word_vector)

            for i in range(len(sentence)):
                relative_vector_entity_a = POS_VECTOR[i - en1_pos, :]
                relative_vector_entity_b = POS_VECTOR[i - en2_pos, :]
                pos_vec = np.concatenate((relative_vector_entity_a, relative_vector_entity_b))
                relative_pos.append(pos_vec)
            if len(sentence_vector) < FIXED_WORD_LENGTH:
                for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                    sentence_vector.append(PLACEHOLDER)
                    pos_vec = np.concatenate((POS_VECTOR[FIXED_WORD_LENGTH, :], POS_VECTOR[FIXED_WORD_LENGTH, :]))
                    relative_pos.append(pos_vec)

            output_sentence.append(sentence_vector)
            output_relation.append(relation)
            output_entity_pos.append([en1_pos, en2_pos])
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

    tag_0 = conc[conc[:, 0] == 0]
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
    tag_11 = conc[conc[:, 0] == 11]
    tag_12 = conc[conc[:, 0] == 12]
    tag_13 = conc[conc[:, 0] == 13]
    tag_14 = conc[conc[:, 0] == 14]
    tag_15 = conc[conc[:, 0] == 15]
    tag_16 = conc[conc[:, 0] == 16]
    tag_17 = conc[conc[:, 0] == 17]
    # tag_18 = conc[conc[:, 0] == 18]

    filter = np.concatenate((
        tag_0, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8, tag_9,
        tag_10, tag_11, tag_12, tag_13, tag_14, tag_15, tag_16, tag_17), axis=0)
    print(filter.shape)

    np.random.shuffle(filter)
    np.save(save_filename, filter)
