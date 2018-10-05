import json
import os
import re
import string
from gensim.models import KeyedVectors
import pyltp
import numpy as np
import pyltp

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
CORPUS_TRAIN = os.path.join(CWD, "corpus_train.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test.txt")

DOC_DIR = "D:\\Projects\\Baike\\parse_data"
LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")
PUNT_PATTERN = re.compile(u'.*?[。|...|？|?|！|!|~|～|#|\n]+')

wordvec = KeyedVectors.load(WORDVEC, mmap='r')
DESC_LENGTH = 80
DIMENSION = 100
PLACEHOLDER = np.zeros(DIMENSION)


def save_desc():
    # see doc2vec.py
    pass


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


# def generate_splited_description():
#     with open('description.json', 'r', encoding="utf8") as fp:
#         desc_dict = json.load(fp)
#     desc_split_dict = {}
#     sentence_list = []
#     for key, value in desc_dict.items():
#         text = desc_preprocess(value)
#         for sentence in PUNT_PATTERN.findall(text + u'#'):  # 分句
#             s = sentence[:-1]  # strip '#'
#             if s.strip() == "":
#                 continue
#             words = segmentor.segment(s)
#             sentence_list.append(list(words))
#         if len(sentence_list) != 0:
#             desc_split_dict[key] = sentence_list
#     print(len(desc_split_dict))
#     with open('description_splited.json', 'w', encoding="utf8") as fp2:
#         json.dump(desc_split_dict, fp2)
# generate_splited_description()
# exit(88)

with open('description.json', 'r', encoding="utf8") as fp:
    desc_dict = json.load(fp)
entity_set = get_entity_set()
entity_desc_dict = {en: desc_preprocess(desc_dict[en]) for en in desc_dict if en in entity_set}
print(len(entity_set), len(entity_desc_dict))

desc_sep_dict = {}
desc_vec_dict = {}

for k, v in entity_desc_dict.items():
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
