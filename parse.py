import re
import os
import json
import sqlite3
import pyltp
import neo4j


def load_relation():
    d = dict()
    with open("person_relation.txt", "r", encoding="utf8") as f:
        for line in f:
            li = line.split(" ")
            # ENTITY_MAP.add(line.split(" ")[0])
            d[li[0]] = (li[1], li[2].rstrip())
    return d


def load_person_entity_set():
    d = set()
    with open("person.txt", "r", encoding="utf8") as f:
        for line in f:
            d.add(line.rstrip())
    return d


def build_relation_pattern(d):
    s = u""
    for k, v in d.items():
        s += k + "|"
    # s = s[0:-1]
    s = s.rstrip('|')
    ptn = u"(" + s + u")"
    return re.compile(ptn)


def parse_content(text, data_dict):
    ret_dict = {}
    ret_sentences = []
    person_entity_links = set()
    # 将内链中所有人物实体取出
    for k, v in data_dict['links'].items():
        if k and k != data_dict['name']:
            lword = [k]
            lpostag = postagger.postag(lword)
            lnetag = recognizer.recognize(lword, lpostag)
            if lnetag[0].endswith("Nh"):
                person_entity_links.add(lword[0])

    for s in PUNT_PATTERN.findall(text + u'#'):  # 分句
        li = RELATION_PATTERN.findall(s)
        for rel in li:
            # 开始命名实体识别
            is_success = False
            # 方式1：依据内链识别
            for k in person_entity_links:
                # if k and k != data_dict['name'] and k in PERSON_ENTITY_SET and k in s and k not in ret_dict[rel]:
                if k in s and (rel not in ret_dict or k not in ret_dict[rel]):  # key在句子中并且key还没存
                    # if k in PERSON_ENTITY_SET:  #对不存在实体库的也包含
                    is_success = True
                    if rel not in ret_dict:
                        ret_dict[rel] = []
                    ret_dict[rel].append(k)
                    ret_sentences.append(s.strip() + " " + rel)
            # 方式2：LTP识别，发现无内链实体，若方式1已发现，则不再进行
            if is_success:
                continue
            words = segmentor.segment(s)
            postags = postagger.postag(words)
            netags = recognizer.recognize(words, postags)
            for i in range(len(netags)):
                # if str(netags[i]).endswith("Nh")  and words[i] != data_dict['name'] and words[i] in PERSON_ENTITY_SET and words[i] not in ret_dict[rel]:
                if str(netags[i]).endswith("Nh") and words[i] != data_dict['name'] and words[i] != rel and (
                # in case: 堂弟:堂弟
                        rel not in ret_dict or words[i] not in ret_dict[rel]):
                    if rel not in ret_dict:
                        ret_dict[rel] = []
                    ret_dict[rel].append(words[i])
                    ret_sentences.append(s.strip() + " " + rel)

    return ret_dict, ret_sentences


def parse_infobox(data_dict):
    ret_dict = {}
    for k, v in data_dict.items():
        if k in RELATION_DICT:
            ret_dict[k] = v
    return ret_dict


LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load(cws_model_path)
postagger = pyltp.Postagger()
postagger.load(pos_model_path)
recognizer = pyltp.NamedEntityRecognizer()
recognizer.load(ner_model_path)

# 开始解析
PARSE_DATA_PATH = "D:\\Projects\\Baike\\parse_data"
RELATION_DICT = load_relation()
PERSON_ENTITY_SET = load_person_entity_set()
RELATION_PATTERN = build_relation_pattern(RELATION_DICT)
PUNT_PATTERN = re.compile(u'.*?[,|，|.|。|...|？|?|！|!|；|;|~|～|#|\n]+')

# DB = neo4j.initDB()

for root, subdirs, files in os.walk(PARSE_DATA_PATH):
    for filename in files:
        if filename != "陈奕迅10.json":
            continue
        file_path = os.path.join(root, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            res_dict1, res_sentence1 = parse_content(data['abstract'], data)
            res_dict2, res_sentence2 = parse_content(data['body'], data)
            res_dict3 = parse_infobox(data['infobox'])
        if len(res_dict1) != 0 or len(res_dict2) != 0 or len(res_dict3) != 0:
            with open("relation_data/" + filename, 'w', encoding='utf8') as g, \
                    open("relation_sentences/" + filename + ".txt", "w", encoding="utf8") as h:
                g.write("###\n")
                g.write(json.dumps(res_dict1, ensure_ascii=False))
                g.write("\n###\n")
                g.write(json.dumps(res_dict2, ensure_ascii=False))
                g.write("\n###\n")
                g.write(json.dumps(res_dict3, ensure_ascii=False))
                h.write("\n".join(res_sentence1))
                h.write("\n")
                h.write("\n".join(res_sentence2))
        # for k, v in res_dict3.items():
        #     neo4j.build_N_R(DB, data['name'], v, RELATION_DICT[k][0], "infobox")
        # for k, v in res_dict1.items():
        #     for person in v:
        #         neo4j.build_N_R(DB, data['name'], person, RELATION_DICT[k][0], "abstract")
        # for k, v in res_dict2.items():
        #     for person in v:
        #         neo4j.build_N_R(DB, data['name'], person, RELATION_DICT[k][0], "body")
# 结束解析
