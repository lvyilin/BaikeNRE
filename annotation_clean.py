# 去掉这种句子：冯少先的儿子中央民族乐团著名中阮演奏家 冯满天（与著名阮乐器演奏家自由音乐人刘星同时学习月琴）     月琴演奏家冯少先 冯少先 冯满天 [1]
# 出现原因是有些实体名为“xxx  [x]”
import os

CWD = os.getcwd()

ANNOTATION_DIR = CWD + "\\sentences_annotation_auto\\"
ANNOTATION_TARGET = ANNOTATION_DIR + "annotation.txt"
ANNOTATION_NEW = ANNOTATION_DIR + "annotation_new.txt"


def split_line(line):
    spl = str(line).rsplit(" ", 3)
    return spl[0], spl[1], spl[2], spl[3]


with open(ANNOTATION_TARGET, "r", encoding="utf8") as f:
    lines = f.readlines()

with open(ANNOTATION_NEW, "w", encoding="utf8") as f:
    for line in lines:
        flag = False
        if line.startswith("@@"):
            flag = True
        else:
            sentence, entity_a, entity_b, relation = split_line(line)
            if '[' not in relation:
                flag = True
        if flag:
            f.write(line)
