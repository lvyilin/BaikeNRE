import mxnet as mx
from gensim.models import KeyedVectors
import numpy as np
import os

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train2.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test2.txt")
DIMENSION = 100
FIXED_WORD_LENGTH = 60

wordvec = KeyedVectors.load(WORDVEC, mmap='r')
PLACEHOLDER = np.zeros(DIMENSION)
POS_VEC = np.random.random((4, DIMENSION))

for corpus, save_filename in ((CORPUS_TRAIN, "data_train_rnn_pi.npy"),
                              (CORPUS_TEST, "data_test_rnn_pi.npy")):
    output_sentence = []
    output_relation = []

    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split()
            entity_a = content[0]
            entity_b = content[1]
            relation = int(content[2])
            sentence = content[3:]

            sentence_vector = []

            for i in range(len(sentence)):
                # if sentence[i] == entity_a:
                #     entity_a_pos = i
                # if sentence[i] == entity_b:
                #     entity_b_pos = i

                if sentence[i] not in wordvec:
                    word_vector = PLACEHOLDER
                    sentence_vector.append(word_vector)
                else:
                    word_vector = wordvec[sentence[i]]
                    if sentence[i] == entity_a:
                        sentence_vector.append(POS_VEC[0])
                        sentence_vector.append(word_vector)
                        sentence_vector.append(POS_VEC[1])
                    elif sentence[i] == entity_b:
                        sentence_vector.append(POS_VEC[2])
                        sentence_vector.append(word_vector)
                        sentence_vector.append(POS_VEC[3])
                    else:
                        sentence_vector.append(word_vector)

            if len(sentence_vector) < FIXED_WORD_LENGTH:
                for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                    sentence_vector.append(PLACEHOLDER)

            output_sentence.append(sentence_vector[:FIXED_WORD_LENGTH])
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
