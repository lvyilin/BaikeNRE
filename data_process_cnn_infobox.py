from gensim.models import KeyedVectors
import numpy as np
import os

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train2.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test2.txt")
DIMENSION = 100
POS_DIMENSION = 5

FIXED_WORD_LENGTH = 60
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
PLACEHOLDER = np.zeros(DIMENSION)
POS_VECTOR = np.random.random((FIXED_WORD_LENGTH * 2, POS_DIMENSION))

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


for corpus, save_filename in ((CORPUS_TRAIN, "data_train_cnn_infobox.npy"),
                              (CORPUS_TEST, "data_test_cnn_infobox.npy")):
    output_sentence = []
    output_relation = []
    output_entity_pos = []
    output_relative_pos = []
    output_en1_infobox = []
    output_en2_infobox = []
    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split()
            entity_a = content[0]
            entity_b = content[1]
            relation = int(content[2])
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
                    word_vector = PLACEHOLDER
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
                    "entity not found: (%s, %d) (%s, %d) @%s" % (
                        entity_a, entity_a_pos, entity_b, entity_b_pos, sentence))
                exception_flag = True
            if entity_a_pos < entity_b_pos:
                entity_pos.append([entity_a_pos, entity_b_pos])
            elif entity_a_pos > entity_b_pos:
                entity_pos.append([entity_b_pos, entity_a_pos])
            else:
                print(
                    "entity equal: (%s, %d) (%s, %d) @%s" % (entity_a, entity_a_pos, entity_b, entity_b_pos, sentence))
                exception_flag = True
                # exit(1)
            if exception_flag:
                if relation == -1:
                    continue
                print(line)
                assert False
            for i in range(len(sentence)):
                relative_vector_entity_a = POS_VECTOR[i - entity_a_pos, :]
                relative_vector_entity_b = POS_VECTOR[i - entity_b_pos, :]
                pos_vec = np.concatenate((relative_vector_entity_a, relative_vector_entity_b))
                relative_pos.append(pos_vec)

            if len(sentence_vector) < FIXED_WORD_LENGTH:
                for i in range(FIXED_WORD_LENGTH - len(sentence_vector)):
                    sentence_vector.append(PLACEHOLDER)
                    pos_vec = np.concatenate((POS_VECTOR[FIXED_WORD_LENGTH, :], POS_VECTOR[FIXED_WORD_LENGTH, :]))
                    relative_pos.append(pos_vec)

            output_sentence.append(sentence_vector)
            output_relation.append(relation)
            output_entity_pos.append(entity_pos)
            output_relative_pos.append(relative_pos)
            output_en1_infobox.append(get_entity_infobox(entity_a))
            output_en2_infobox.append(get_entity_infobox(entity_b))

    print("length of output_sentence: %d" % len(output_sentence))

    np_sentence = np.array(output_sentence, dtype=float)
    np_relation = np.array(output_relation, dtype=int)
    np_entity_pos = np.array(output_entity_pos, dtype=int)
    np_relative_pos = np.array(output_relative_pos, dtype=float)
    np_en1_infobox = np.array(output_en1_infobox, dtype=float)
    np_en2_infobox = np.array(output_en2_infobox, dtype=float)

    print(np_sentence.shape)
    print(np_relative_pos.shape)
    print(np_entity_pos.shape)
    print(np_en1_infobox.shape)
    print(np_en2_infobox.shape)

    np_sentence_matrix = np.concatenate((np_sentence, np_relative_pos), axis=2)
    print(np_sentence_matrix.shape)

    sentence_vec = np_sentence_matrix.reshape(np_sentence_matrix.shape[0],
                                              (DIMENSION + 2 * POS_DIMENSION) * FIXED_WORD_LENGTH)
    entity_pos_vec = np_entity_pos.reshape(np_entity_pos.shape[0], 2)
    np_en_infobox = np.concatenate((np_en1_infobox.reshape(np_en1_infobox.shape[0], -1),
                                    np_en2_infobox.reshape(np_en2_infobox.shape[0], -1)), axis=1)
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
