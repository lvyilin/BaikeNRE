# 解析出主体与另外实体共现的句子，主体为词条名
import json
import os
import pyltp
import re

CWD = os.getcwd()
SAVE_DIR = CWD + "\\entity_sentences\\"

LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")
postagger = pyltp.Postagger()
postagger.load(pos_model_path)
recognizer = pyltp.NamedEntityRecognizer()
recognizer.load(ner_model_path)

PARSE_DATA_PATH = "D:\\Projects\\Baike\\parse_data"
PUNT_PATTERN = re.compile(u'.*?[,|，|.|。|...|？|?|！|!|；|;|~|～|#|\n]+')


def parse_content(text, name, ret_list):
    for sentence in PUNT_PATTERN.findall(text + u'#'):  # 分句
        if name not in sentence:
            continue
        words = segmentor.segment(sentence)
        postags = postagger.postag(words)
        netags = recognizer.recognize(words, postags)
        for i in range(len(netags)):
            if str(netags[i]).endswith("Nh") and words[i] != name:
                item = (name, words[i], sentence)
                ret_list.append(item)
    return


for root, subdirs, files in os.walk(PARSE_DATA_PATH):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            ret_list = []
            data = json.load(f)
            parse_content(data['abstract'], data['name'], ret_list)
            parse_content(data['body'], data['name'], ret_list)

            list_len = len(ret_list)
            if list_len != 0:
                print(filename)
                with open(SAVE_DIR + filename, "w", encoding="utf-8") as g:
                    for i in range(list_len):
                        g.write("{} {} {}\n".format(ret_list[i][0], ret_list[i][1], ret_list[i][2]))
