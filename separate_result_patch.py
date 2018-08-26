import os

IS_FOR_NO_RELATION = False

CWD = os.getcwd()

SEP_CORPUS = os.path.join(CWD, "separated_corpus.txt")
SEP_CORPUS_LABELED = os.path.join(CWD, "separated_corpus_with_label.txt")
if IS_FOR_NO_RELATION:
    SEP_CORPUS_LABELED = os.path.join(CWD, "separated_corpus_with_label_no_rel.txt")

NEW_FILE = os.path.join(CWD, "separated_corpus_patch.txt")
NEW_LABELED_FILE = os.path.join(CWD, "separated_corpus_with_label_patch.txt")
if IS_FOR_NO_RELATION:
    NEW_LABELED_FILE = os.path.join(CWD, "separated_corpus_with_label_patch_no_rel.txt")

PATCH = CWD + "\\separate_result_patch.txt"

old_str = []
new_str = []
with open(PATCH, "r", encoding="utf8") as f:
    for line in f:
        content = line.strip().split("\t")
        old_str.append(content[0])
        new_str.append(content[1])

with open(SEP_CORPUS, "r", encoding="utf8") as g, open(NEW_FILE, "w", encoding="utf8")as h:
    text = g.read()
    for i in range(len(old_str)):
        text = text.replace(old_str[i], new_str[i])
    h.write(text)

with open(SEP_CORPUS_LABELED, "r", encoding="utf8") as g, open(NEW_LABELED_FILE, "w", encoding="utf8")as h:
    text = g.read()
    for i in range(len(old_str)):
        text = text.replace(old_str[i], new_str[i])
    h.write(text)
