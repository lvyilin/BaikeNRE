import os
import sqlite3

import numpy as np
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train_SemEval.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test_SemEval.txt")
DIMENSION = 100
POS_DIMENSION = 5
FIXED_WORD_LENGTH = 60
MAX_ENTITY_DEGREE = 50
ENTITY_DEGREE = MAX_ENTITY_DEGREE + 1
PLACEHOLDER = np.zeros(DIMENSION)
wordvec = KeyedVectors.load(WORDVEC, mmap='r')

entity_dict = {}

for corpus in (CORPUS_TRAIN, CORPUS_TEST):
    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split("\t")
            entity_a = content[1]
            entity_b = content[2]
            if entity_a not in entity_dict:
                entity_dict[entity_a] = []
            if entity_b not in entity_dict:
                entity_dict[entity_b] = []
            if entity_b not in entity_dict[entity_a]:
                entity_dict[entity_a].append(entity_b)
            if entity_a not in entity_dict[entity_b]:
                entity_dict[entity_b].append(entity_a)

entity_len = [len(x) for x in entity_dict.values()]
print(len(entity_len))
plt.hist(entity_len, bins='auto')
plt.show()

entityvec_dict = {}
for entity, neighbor_entity_set in entity_dict.items():
    neighbor_entity_list = list(neighbor_entity_set)
    if len(neighbor_entity_list) > MAX_ENTITY_DEGREE:
        print("over max degree: %s" % entity)
        neighbor_entity_list[:] = neighbor_entity_list[:MAX_ENTITY_DEGREE]
    if entity not in wordvec:
        print(entity)
        continue
    output_entity_vec = []
    entity_vec = wordvec[entity]
    output_entity_vec.append(entity_vec)
    for neighbor_entity in neighbor_entity_list:
        neighbor_vec = wordvec[neighbor_entity] if neighbor_entity in wordvec else PLACEHOLDER
        output_entity_vec.append(neighbor_vec)
    output_edge_vec = []
    for i in range(len(output_entity_vec)):
        # edge_vec =  neigh + self
        edge_vec = np.concatenate((output_entity_vec[i], entity_vec))
        output_edge_vec.append(edge_vec)

    mask = np.concatenate((np.ones(len(output_edge_vec)), np.zeros(ENTITY_DEGREE - len(output_edge_vec))))
    if len(output_edge_vec) < ENTITY_DEGREE:
        for i in range(ENTITY_DEGREE - len(output_edge_vec)):
            output_edge_vec.append(np.zeros(output_edge_vec[0].shape))
    np_output_edge_vec = np.array(output_edge_vec, dtype=float).reshape(-1)
    # mask + edge_vec
    entityvec_dict[entity] = np.concatenate((mask, np_output_edge_vec))

with open("entity2vec_SemEval_key.txt", "w", encoding="utf8") as f:
    for k in entityvec_dict.keys():
        f.write(k + "\n")

values = np.array(list(entityvec_dict.values()), dtype=float)
np.save("entity2vec_SemEval_value.npy", values)
