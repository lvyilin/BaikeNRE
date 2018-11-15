import json
import os

import numpy as np
from gensim.models import KeyedVectors

DIMENSION = 100
INFOBOX_VALUE_LENGTH = 10
INFOBOX_LENGTH = 20
CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
PLACEHOLDER = np.zeros(DIMENSION)
PLACEHOLDER_INFOBOX = [PLACEHOLDER] * INFOBOX_VALUE_LENGTH
infoboxvec_dict = {}
with open("corpus_infobox.json", "r", encoding="utf8") as fp:
    infobox_dict = json.load(fp)

for key, val in infobox_dict.items():
    if key not in wordvec:
        continue
    infobox = []
    for k, v in val.items():
        if k in wordvec:
            infobox_vals = [wordvec[k]]
        else:
            infobox_vals = [PLACEHOLDER]
        for w in v:
            infobox_vals.append(wordvec[w])
        if len(infobox_vals) < INFOBOX_VALUE_LENGTH:
            for i in range(INFOBOX_VALUE_LENGTH - len(infobox_vals)):
                infobox_vals.append(PLACEHOLDER)
        infobox_vals = infobox_vals[:INFOBOX_VALUE_LENGTH]
        infobox.append(infobox_vals)
    if len(infobox) < INFOBOX_LENGTH:
        for i in range(INFOBOX_LENGTH - len(infobox)):
            infobox.append(PLACEHOLDER_INFOBOX)
    infobox = infobox[:INFOBOX_LENGTH]
    infobox_vec = np.array(infobox)
    assert infobox_vec.shape == (INFOBOX_LENGTH, INFOBOX_VALUE_LENGTH, DIMENSION)
    infoboxvec_dict[key] = infobox_vec

with open("infobox2vec_key.txt", "w", encoding="utf8") as f:
    for k in infoboxvec_dict.keys():
        f.write(k + "\n")

values = np.array(list(infoboxvec_dict.values()), dtype=float)
np.save("infobox2vec_value.npy", values)
