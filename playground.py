# encoding=utf-8
import urllib.request
from bs4 import BeautifulSoup
import json
import os
import re
import pymongo
import shutil

def inf_table(bs, d):
    str_list = []
    i = 0
    for k in bs.find_all('table'):
        for tr in k.find_all('tr'):
            str_list.append(str(tr.get_text()).strip())
    d['table'] = str_list.copy()


def inf_infocox(bs, d):
    i = 1
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
    d['body'] = ""
    for k in bs.find_all('div', class_='para'):
        if str(k.parent.get("class")) == "['lemma-summary']":
            continue
        d['body'].join(str(k.get_text()).strip())
    # d['body'] = ''.join(str_list)


def inf_summary(bs, d):
    str_list = []
    for k in bs.find_all('div', class_='lemma-summary'):
        str_list.append(str(k.get_text()).strip())
    d['abstract'] = ''.join(str_list)


def inf_name(bs, d):
    title_node = bs.find('dd', class_="lemmaWgt-lemmaTitle-title").find("h1")
    if title_node is not None:
        d['name'] = str(title_node.get_text().strip())


def inf_lables(bs, d):
    for labels in bs.find_all("span", class_="taglist"):
        d["tags"].append(str(labels.get_text()).strip())


def write_json(array):
    with open("test.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(array, ensure_ascii=False))


def load_entity():
    d = dict()
    with open("person_relation.txt", "r", encoding="utf8") as f:
        for line in f:
            li = line.split(" ")
            # ENTITY_MAP.add(line.split(" ")[0])
            d[li[0]] = [li[1], li[2]]
    return d


def build_relation_pattern(d):
    s = u""
    for k, v in d.items():
        s += k + "|"
    # s = s[0:-1]
    s = s.rstrip('|')
    ptn = u"(" + s + u")"
    return re.compile(ptn)


dataset_path = "D:\\result\\"
# for filename in os.listdir(dataset_path):
with open("person.txt", "a", encoding="utf-8")as person_file:
    tillLastErrorPos = False
    for root, subdirs, files in os.walk(dataset_path):
        cnt = 0
        for filename in files:
            if not tillLastErrorPos and filename != u"张伟210":
                continue
            else:
                tillLastErrorPos = True
            if filename.endswith(".jpg"):
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
                    # inf_name(bs, result_dict)
                    result_dict['name'] = filename
                except Exception as e:
                    print("error:"+filename)
                    with open("error.txt","a",encoding="utf-8") as k:
                        k.write(filename+"\n")

                save_name = filename

                person_file.write(save_name+"\n")
                print(filename)
                cnt = cnt + 1
        # if cnt > 10:
        #     break
