import pyltp
import re

from gensim.models import KeyedVectors
import json
import os

CWD = os.getcwd()
WORDVEC = os.path.join(CWD, "wordvectors.kv")
wordvec = KeyedVectors.load(WORDVEC, mmap='r')
DOC_DIR = "D:\\Projects\\Baike\\parse_data"
LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")

PARSE_DATA_DIR = "D:\\Projects\\Baike\\parse_data"
punc_pattern = re.compile('[,.?!，。？！；：“”"（）()、【】&《》]')
refer_pattern = re.compile('\[\d+\]')
infobox_dict = {}
for filename in os.listdir(PARSE_DATA_DIR):
    with open(os.path.join(PARSE_DATA_DIR, filename), 'r', encoding='utf8') as fp:
        jsond = json.load(fp)
        infobox_dict[jsond['name']] = jsond['infobox']

val_lens = []
maxlen = 0
for key, val in list(infobox_dict.items()):
    for k, v in list(val.items()):
        sen = re.sub(refer_pattern, '', v)
        sen = re.sub(punc_pattern, ' ', sen)
        words = segmentor.segment(sen)
        val_list = []
        for w in words:
            w = w.strip()
            if w != '' and w in wordvec:
                val_list.append(w)
        if len(val_list) > maxlen:
            maxlen = len(val_list)
            print(' '.join(val_list))
        if len(val_list) == 0:
            del val[k]
        else:
            val[k] = val_list
            val_lens.append(len(val_list))
    if len(val) == 0:
        del infobox_dict[key]
    else:
        infobox_dict[key] = val

print("Max value length: %d" % max(val_lens))
with open("corpus_infobox.json", "w", encoding="utf8") as fp:
    json.dump(infobox_dict, fp, ensure_ascii=False)
