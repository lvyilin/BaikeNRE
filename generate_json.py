# encoding=utf-8
import urllib.request
from bs4 import BeautifulSoup
import json
import os
import re
import string
import pymongo
import shutil
import pyltp


def inf_table(bs, d):
    str_list = []
    i = 0
    for k in bs.find_all('table'):
        for tr in k.find_all('tr'):
            str_list.append(str(tr.get_text()).strip())
    d['table'] = str_list.copy()


def inf_infocox(bs, d):
    i = 1
    for k in bs.find_all('div', class_ = 'star-info-block relations'):
        for tr in k.find_all('div', class_ = 'name'):
            d['infobox'][str(tr.contents[0]).strip()] = str(tr.contents[1]).strip().replace('<em>','').replace('</em>','')

    for k in bs.find_all('div', class_='basic-info cmn-clearfix'):
        for tr in k.find_all('dt', class_="basicInfo-item name"):
            temp = 1
            for tt in k.find_all('dd', class_="basicInfo-item value"):
                if (temp != i):
                    temp = temp + 1
                    continue
                tr_txt = tr.get_text().replace(u"\xa0", "").replace(".", "").strip()
                d['infobox'][tr_txt] = str(tt.get_text()).strip()
                break
            i = i + 1


def inf_para(bs, d):
    str_list = []
    for k in bs.find_all('div', class_='para'):
        if str(k.parent.get("class")) == "['lemma-summary']":
            continue
        str_list.append(str(k.get_text()).replace('\n','').replace('\xa0','') + "\n")
    d['body'] = ''.join(str_list)


# def inf_summary(bs, d):
#     str_list = []
#     for k in bs.find_all('div', class_='lemma-summary'):
#         str_list.append(str(k.get_text()).strip())
#     d['abstract'] = ''.join(str_list)

def inf_summary(bs, d):
    str_list = []
    for k in bs.find_all('div', class_='lemma-summary'):
        for j in k.find_all('div', class_='para'):
            str_list.append(str(j.get_text()).replace('\n','').replace('\xa0','') + '\n')
    d['abstract'] = ''.join(str_list)


def inf_name(bs, d):
    title_node = bs.find('dd', class_="lemmaWgt-lemmaTitle-title")
    if title_node is not None:
        title_node = title_node.find("h1")
    if title_node is not None:
        d['name'] = str(title_node.get_text().strip())


def inf_lables(bs, d):
    for labels in bs.find_all("span", class_="taglist"):
        d["tags"].append(str(labels.get_text()).strip())


def inf_links(bs, d):
    links = {}
    for link in bs.find_all('a', href=re.compile(r"/item/")):
        links[str(link.get_text()).strip()] = str(link['href'])
    d["links"] = links


def write_json(resdict, filename):
    with open("parse_data/" + filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(resdict, ensure_ascii=False))


def load_entity():
    d = dict()
    with open("person_relation.txt", "r", encoding="utf8") as f:
        for line in f:
            li = line.split(" ")
            # ENTITY_MAP.add(line.split(" ")[0])
            d[li[0]] = (li[1], li[2])
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


DATASET_PATH = "D:\\result\\"
# LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
# cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
# ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
#
# segmentor = pyltp.Segmentor()
# segmentor.load(cws_model_path)
# postagger = pyltp.Postagger()  # 初始化实例
# postagger.load(pos_model_path)
# recognizer = pyltp.NamedEntityRecognizer()  # 初始化实例
# recognizer.load(ner_model_path)  # 加载模型
person_entity_set = load_person_entity_set()
for root, subdirs, files in os.walk(DATASET_PATH):
    for filename in files:
        if filename.endswith(".jpg"):
            continue
        if filename!='杨开慧46':
            continue
        if filename not in person_entity_set:
            continue
        if os.path.isfile("parse_data/" + filename + ".json"):
            continue
        file_path = os.path.join(root, filename)

        with open(file_path, "r", encoding="utf8") as f:
            result_dict = {
                'name': "",
                'tags': [],
                'abstract': "",
                'infobox': {},
                'body': "",
                'table': []
            }
            try:
                bs = BeautifulSoup(f, "lxml")

                inf_lables(bs, result_dict)
                is_person_entity = False
                for tag in result_dict['tags']:
                    if str(tag).endswith(u"人物"):
                        is_person_entity = True
                        break
                if not is_person_entity:
                    continue
                inf_name(bs, result_dict)
                if result_dict['name'] == "":
                    result_dict['name'] = filename.rstrip(string.digits)
                inf_summary(bs, result_dict)
                inf_para(bs, result_dict)
                inf_infocox(bs, result_dict)
                inf_table(bs, result_dict)
                inf_links(bs, result_dict)

            except:
                print("error:" + filename)
                continue

            save_name = filename
            # if os.path.exists(save_name + ".json"):
            #     i = 1
            #     while os.path.exists(save_name + i + ".json"):
            #         i += 1
            #     save_name = save_name + i
            write_json(result_dict, save_name + ".json")

