# 为分词后的每一句子添加ID，用于获得失败结果
import os

CWD = os.getcwd()
CORPUS = os.path.join(CWD, "separated_corpus_with_label_patch_amend.txt")
ID_CORPUS = os.path.join(CWD, "separated_corpus_with_label_patch_amend_id.txt")
FAIL_ID = os.path.join(CWD, "fail_id_cnssnn.txt")
FAIL_SENTENCE = os.path.join(CWD, "fail_sentence_cnssnn.txt")


def insert_id():
    line_count = 0
    with open(CORPUS, "r", encoding="utf8") as f, open(ID_CORPUS, "w", encoding="utf8") as g:
        for line in f:
            g.write("%d %s" % (line_count, line))
            line_count += 1


def map_id_to_sentence():
    min_length = 100000
    min_list = []
    with open(FAIL_ID, "r") as f_id:
        while True:
            line = f_id.readline().strip()
            if line == "":
                break
            content = line.split(" ")
            if len(content) < min_length:
                min_length = len(content)
                min_list = content
    sentences = []
    id_list = [int(x) for x in min_list]
    id_set = set(id_list)
    with open(ID_CORPUS, "r", encoding="utf8") as f_corpus, open(FAIL_SENTENCE, "w", encoding="utf8")as f_sentence:
        for line in f_corpus:
            content = line.split(" ", maxsplit=2)
            this_id = int(content[0])
            if this_id in id_set:
                f_sentence.write(line)


insert_id()
# map_id_to_sentence()
