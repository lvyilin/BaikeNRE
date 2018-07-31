# 将之前匹配方法标注的结果结合到共现标注的结果的数据库里
import sqlite3

ANNOTATION_OLD_FILE = "D:\\Projects\\Baike\\sentences_annotation_auto\\annotation_fin.txt"
PERSON_RELATION_FILE = "D:\\Projects\\Baike\\person_relation.txt"
RELATION_MAP_FILE = "D:\\Projects\\Baike\\relation_stat_for_map.txt"
DB = "D:\\Projects\\Baike\\baike.db"


def split_line(line):
    spl = str(line).rsplit(" ", 3)
    return spl[0], spl[1], spl[2], spl[3]


def load_relation():
    d = dict()
    with open(PERSON_RELATION_FILE, "r", encoding="utf8") as f:
        for line in f:
            li = line.split(" ")
            d[li[0]] = li[1]
    return d


def load_relation_map():
    d_num = dict()
    with open(RELATION_MAP_FILE, "r", encoding="utf8") as f:
        i = 1
        for line in f:
            lst = line.split("：")
            relation_type = lst[0]
            relation_name = lst[1].split(" ")
            for rel in relation_name:
                rel = rel.strip()
                d_num[rel] = i
            i += 1
    return d_num


RELATION_NAME = load_relation()
RELATION_NUM_MAP = load_relation_map()


def map_relation_to_num(relation):
    if relation in RELATION_NAME:
        rel = RELATION_NAME[relation]
        print("{},{},{}".format(relation, rel, RELATION_NUM_MAP[rel]))
        return RELATION_NUM_MAP[rel]


with open(ANNOTATION_OLD_FILE, "r", encoding="utf8") as f:
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    for line in f:
        sentence, entity_a, entity_b, relation = split_line(line)
        relation = relation.strip()
        c.execute(
            "select * from Data where relation!=8 and sentence='{}' and entity_a='{}' and entity_b='{}'".format(
                sentence, entity_a, entity_b))
        for row in c:
            print(row)
            r = map_relation_to_num(relation)
            if len(row) != 0 and r is not None:
                id = row[0]
                c.execute("update Data set relation={} where id={}".format(r, id))
            else:
                print(row)
    conn.commit()
    conn.close()
