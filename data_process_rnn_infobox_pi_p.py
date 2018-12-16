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

infobox_key = []
infobox_value = np.load('infobox2vec_value.npy')
with open("infobox2vec_key.txt", "r", encoding="utf8") as f:
    for line in f:
        infobox_key.append(line.strip())


def get_entity_infobox(entity_name):
    if entity_name in infobox_key:
        idx = infobox_key.index(entity_name)
        return infobox_value[idx]
    else:
        return np.zeros(infobox_value[0].shape)


for corpus, save_filename in ((CORPUS_TRAIN, "data_train_rnn_infobox_pi_p.npy"),
                              (CORPUS_TEST, "data_test_rnn_infobox_pi_p.npy")):
    output_sentence = []
    output_relation = []
    output_en1_infobox = []
    output_en2_infobox = []
    output_entity_pos = []

    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split()
            entity_a = content[0]
            entity_b = content[1]
            relation = int(content[2])
            sentence = content[3:]

            sentence_vector = []
            entity_pos = []
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

            d_pos = FIXED_WORD_LENGTH
            for i in entity_a_pos_list:
                for j in entity_b_pos_list:
                    if abs(i - j) < d_pos:
                        d_pos = abs(i - j)
                        entity_a_pos = i
                        entity_b_pos = j
            if entity_a_pos < entity_b_pos:
                entity_pos.append([entity_a_pos, entity_b_pos])
            elif entity_a_pos > entity_b_pos:
                entity_pos.append([entity_b_pos, entity_a_pos])

            if len(sentence_vector) < FIXED_WORD_LENGTH:
                for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                    sentence_vector.append(PLACEHOLDER)

            output_sentence.append(sentence_vector[:FIXED_WORD_LENGTH])
            output_relation.append(relation)
            output_en1_infobox.append(get_entity_infobox(entity_a))
            output_en2_infobox.append(get_entity_infobox(entity_b))
            output_entity_pos.append(entity_pos)

    print("length of output_sentence: %d" % len(output_sentence))

    np_sentence = np.array(output_sentence, dtype=float)
    np_relation = np.array(output_relation, dtype=int)
    np_en1_infobox = np.array(output_en1_infobox, dtype=float)
    np_en2_infobox = np.array(output_en2_infobox, dtype=float)
    np_entity_pos = np.array(output_entity_pos, dtype=int)

    print(np_sentence.shape)
    print(np_en1_infobox.shape)
    print(np_en2_infobox.shape)

    sentence_vec = np_sentence.reshape(np_sentence.shape[0],
                                       DIMENSION * FIXED_WORD_LENGTH)
    np_en_infobox = np.concatenate((np_en1_infobox.reshape(np_en1_infobox.shape[0], -1),
                                    np_en2_infobox.reshape(np_en2_infobox.shape[0], -1)), axis=1)
    entity_pos_vec = np_entity_pos.reshape(np_entity_pos.shape[0], 2)

    # relation + sentence_vec
    conc = np.concatenate((np.expand_dims(np_relation, axis=1),
                           entity_pos_vec,
                           sentence_vec,
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
