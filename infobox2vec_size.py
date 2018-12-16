# 添加可INFOBOX实际尺寸，用于可变卷积核
import json
import os
import pickle
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

infobox_size_dict = {}
for key, val in infobox_dict.items():
    if key not in wordvec:
        continue
    infobox = []
    val_size = []
    infobox_count = 0
    for k, v in val.items():
        if infobox_count >= INFOBOX_LENGTH:
            break
        if k in wordvec:
            infobox_vals = [wordvec[k]]
        else:
            infobox_vals = [PLACEHOLDER]

        infobox_vals_count = 1
        for w in v:
            if infobox_vals_count >= INFOBOX_VALUE_LENGTH:
                break
            infobox_vals.append(wordvec[w])
            infobox_vals_count += 1

        val_size.append(infobox_vals_count)
        infobox += infobox_vals
        infobox_count += 1
    infobox_size_dict[key] = val_size

    infobox_vec = np.array(infobox)
    assert infobox_vec.shape[1] == DIMENSION
    result = np.zeros((INFOBOX_VALUE_LENGTH * INFOBOX_LENGTH, DIMENSION))
    result[:infobox_vec.shape[0], :infobox_vec.shape[1]] = infobox_vec
    # assert infobox_vec.shape == (INFOBOX_LENGTH, INFOBOX_VALUE_LENGTH, DIMENSION)
    infoboxvec_dict[key] = result

with open("infobox2vec_size_key.txt", "w", encoding="utf8") as f:
    for k in infoboxvec_dict.keys():
        f.write(k + "\n")
with open("infobox2vec_size.txt", "wb") as f:
    pickle.dump(infobox_size_dict, f)

values = np.array(list(infoboxvec_dict.values()), dtype=float)
np.save("infobox2vec_size_value.npy", values)
