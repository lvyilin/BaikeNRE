import json
import os
import pyltp
import re

import numpy as np
from gensim.models import KeyedVectors

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test.txt")
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
    print(len(desc_dict))
    with open('description.json', 'w', encoding="utf8") as fp:
        json.dump(desc_dict, fp)


def get_entity_set():
    entity_set = set()
    for corpus in (CORPUS_TRAIN, CORPUS_TEST):
        with open(corpus, "r", encoding="utf8") as f:
            for line in f:
                content = line.strip().split()
                entity_a = content[0]
                entity_b = content[1]
                for entity in (entity_a, entity_b):
                    if entity in wordvec:
                        entity_set.add(entity)
    return entity_set


def desc_preprocess(desc: str) -> str:
    reference_pattern = re.compile("\[\d+\]")

    def process(desc):
        new_desc = re.sub(reference_pattern, " ", desc)
        return new_desc

    return process(desc)


with open('description.json', 'r', encoding="utf8") as fp:
    desc_dict = json.load(fp)
entity_set = get_entity_set()
entity_desc_dict = {en: desc_preprocess(desc_dict[en]) for en in desc_dict if en in entity_set}
print(len(entity_set), len(entity_desc_dict))

LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")

for key in list(entity_desc_dict.keys()):
    words = list(segmentor.segment(entity_desc_dict[key]))
    if len(words) is 0:
        del entity_desc_dict[key]
        continue
    entity_desc_dict[key] = words

with open("corpus_description.json", "w", encoding="utf8") as gp:
    json.dump(entity_desc_dict, gp)

# desc_key = list(entity_desc_dict.keys())
# corpus_input = [TaggedDocument(v, k) for k, v in entity_desc_dict.items()]
# model = Doc2Vec(corpus_input, vector_size=100, window=5, min_count=3, workers=8, epochs=100)
# model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
#
# # with open("doc2vec_key.txt", "w", encoding="utf8") as f:
# #     f.write("\n".join(desc_key))
# model.wv.save("doc2vectors.kv")
