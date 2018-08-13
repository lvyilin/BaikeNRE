import os

CWD = os.getcwd()
CORPUS = CWD + "\\separated_corpus_with_label_patch.txt"
ID_CORPUS = CWD + "\\separated_corpus_with_label_patch_id.txt"
line_count = 0
with open(CORPUS, "r", encoding="utf8") as f, open(ID_CORPUS, "w", encoding="utf8") as g:
    for line in f:
        g.write("%d %s" % (line_count, line))
        line_count += 1
