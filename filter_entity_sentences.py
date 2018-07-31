import os
import re

FILE = "D:\\Projects\\Baike\\entity_sentences_lite.txt"
NEW_FILE = "D:\\Projects\\Baike\\entity_sentences_v7.txt"


def load_relation():
    d = set()
    with open("person_relation.txt", "r", encoding="utf8") as f:
        for line in f:
            li = line.split(" ")
            # ENTITY_MAP.add(line.split(" ")[0])
            d.add(li[0])
    return d


def build_relation_pattern(d):
    s = u""
    for k in d:
        s += k + "|"
    s = s.rstrip('|')
    ptn = u"(" + s + u")"
    return re.compile(ptn)


def split_sentence(line):
    spl = str(line).split(" ", 2)
    return spl[0], spl[1], spl[2]


RELATION_DICT = load_relation()
RELATION_PATTERN = build_relation_pattern(RELATION_DICT)

with open(FILE, 'r', encoding="utf8") as f:
    lines = f.readlines()
with open(NEW_FILE, "w", encoding="utf8") as f:
    for line in lines:
        # print(len(line))
        # if len(line) >= 175 and len(RELATION_PATTERN.findall(line)) == 0:
        entity_a, entity_b, sentence = split_sentence(line.strip())
        if "'" == entity_a or "'" == entity_b:
            print(line)
            continue
        f.write(line)
