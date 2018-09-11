import json
import os
import string
from gensim.models import KeyedVectors
import pyltp
import numpy as np

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS = os.path.join(CWD, "separated_corpus_with_label_patch_amend_id.txt")
DOC_DIR = "D:\\Projects\\Baike\\parse_data"
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
DESC_LENGTH = 80
DIMENSION = 100
PLACEHOLDER = np.zeros(DIMENSION)


def save_desc():
    desc_dict = {}
    for filename in os.listdir(DOC_DIR):
        with open(os.path.join(DOC_DIR, filename), "r", encoding="utf8") as f:
            jsond = json.load(f)
            name = jsond["name"]
            abstract = jsond["abstract"]
            desc_dict[name] = abstract

    entity_set = set()
    with open(CORPUS, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split()
            entity_a = content[1]
            entity_b = content[2]
            for entity in (entity_a, entity_b):
                if entity in wordvec:
                    entity_set.add(entity)

    print(len(desc_dict))
    en_desc_dict = {en: desc_dict[en] for en in desc_dict if en in entity_set}
    print(len(en_desc_dict))
    with open('description.json', 'w', encoding="utf8") as fp:
        json.dump(en_desc_dict, fp)


with open('description.json', 'r', encoding="utf8") as fp:
    desc_dict = json.load(fp)

LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")

desc_sep_dict = {}
desc_vec_dict = {}

for k, v in desc_dict.items():
    words = segmentor.segment(v)
    if len(words) is 0:
        continue
    desc_sep_dict[k] = list(words[:80])
    desc_vec = []
    for word in desc_sep_dict[k]:
        if word in wordvec:
            desc_vec.append(wordvec[word])
        else:
            desc_vec.append(PLACEHOLDER)
    if len(desc_vec) < DESC_LENGTH:
        for i in range(DESC_LENGTH - len(desc_vec)):
            desc_vec.append(PLACEHOLDER)
    np_desc_vec = np.array(desc_vec)
    desc_vec_dict[k] = np_desc_vec
with open("desc2vec_key.txt", "w", encoding="utf8") as f:
    f.write("\n".join(list(desc_vec_dict.keys())))
values = np.array(list(desc_vec_dict.values()), dtype=float)
np.save("desc2vec_value.npy", values)
