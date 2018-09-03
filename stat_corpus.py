import os
import sqlite3

import numpy as np
from gensim.models import KeyedVectors

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")

CORPUS_TRAIN = os.path.join(CWD, "corpus_train.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test.txt")
conn = sqlite3.connect('baike.db')
c = conn.cursor()
DIMENSION = 100
POS_DIMENSION = 5
FIXED_WORD_LENGTH = 60
MAX_ENTITY_DEGREE = 50
ENTITY_DEGREE = MAX_ENTITY_DEGREE + 1
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
wordvec['UNK'] = np.zeros(DIMENSION)
wordvec['BLANK'] = np.zeros(DIMENSION)
# entity_map = {}
# count = [0 for x in range(11)]
# for corpus in (CORPUS_TRAIN, CORPUS_TEST):
#     with open(corpus, "r", encoding="utf8") as f:
#         for line in f:
#             content = line.strip().split()
#             entity_a = content[0]
#             entity_b = content[1]
#             relation = int(content[2])
#             sentence = content[3:]
#
#             if relation in (2,4,6,10):
#                 relation = 1
#             elif relation is 8:
#                 relation  = 4
#             elif relation is -1:
#                 relation = 0
#
#             count[relation] +=1
# print(count)
# print(max(entity_map.values()))
# count = [0 for x in range(11)]

entity_set = set()
for corpus in (CORPUS_TRAIN, CORPUS_TEST):
    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split()
            entity_a = content[0]
            entity_b = content[1]
            relation = int(content[2])
            sentence = content[3:]

            entity_set.add(entity_a)
            entity_set.add(entity_b)
print(len(entity_set))

entity_edge_map = {}
for entity in entity_set:
    output_entity_vec = []
    if entity not in wordvec:
        continue
    entity_vec = wordvec[entity]
    neighbor_entity_set = set()
    c.execute(
        "select entity_b from Data where entity_a=? union select entity_b from Data3 where entity_a=? GROUP BY entity_b",
        (entity, entity))
    for row in c:
        neighbor_entity_set.add(row[0])
    c.execute(
        "select entity_a from Data where entity_b=? union select entity_a from Data3 where entity_b=? GROUP BY entity_a ",
        (entity, entity))
    for row in c:
        neighbor_entity_set.add(row[0])

    if len(neighbor_entity_set) > MAX_ENTITY_DEGREE:
        continue

    neighbor = []
    for neighbor_entity in neighbor_entity_set:
        if neighbor_entity in wordvec:
            neighbor.append(neighbor_entity)
    entity_edge_map[entity] = neighbor

print(len(entity_edge_map))
val_count = [len(x) for x in entity_edge_map.values()]
print(max(val_count))
print(min(val_count))
print(sum(val_count))
print(sum(val_count) / len(val_count))
