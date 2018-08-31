import os
import random

CWD = os.getcwd()
CORPUS = CWD + "\\separated_corpus_with_label_patch_amend_id.txt"
TRAIN_CORPUS = CWD + "\\corpus_train.txt"
TEST_CORPUS = CWD + "\\corpus_test.txt"
TRAIN_CORPUS_ID = CWD + "\\corpus_train_id.txt"
TEST_CORPUS_ID = CWD + "\\corpus_test_id.txt"

TRAIN_RADIO = 0.7

tag_list = []
for i in range(11):
    tag_list.append([])
with open(CORPUS, "r", encoding="utf8") as f:
    for line in f:
        content = line.strip().split()
        relation = int(content[3])
        if relation == -1:
            relation = 0
        tag_list[relation].append(line)

for i in range(11):
    random.shuffle(tag_list[i])

train_list = [x[:int(len(x) * TRAIN_RADIO)] for x in tag_list]
test_list = [x[int(len(x) * TRAIN_RADIO):] for x in tag_list]

for data, save_file, save_file_id in (
        (train_list, TRAIN_CORPUS, TRAIN_CORPUS_ID), (test_list, TEST_CORPUS, TEST_CORPUS_ID)):
    with open(save_file_id, "w", encoding="utf8") as f, open(save_file, "w", encoding="utf8") as g:
        total_list = []
        for tag in data:
            total_list += tag
        random.shuffle(total_list)
        f.write("".join(total_list))
        total_list_no_id = []
        for line in total_list:
            content = line.split(" ", maxsplit=1)
            sentence = content[1]
            total_list_no_id.append(sentence)
        g.write("".join(total_list_no_id))
