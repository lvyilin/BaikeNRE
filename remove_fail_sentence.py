import os

CWD = os.getcwd()

sen = []
with open(CWD + "\\aaa.txt", "r", encoding="utf8") as f:
    for line in f:
        content = line.split(" ", maxsplit=1)
        idx = content[0]
        sentence = content[1:]
        sen.append("".join(sentence))

with open(CWD + "\\separated_corpus_with_label_patch_amend.txt", "r", encoding="utf8") as f, open(
        CWD + "\\separated_corpus_with_label_patch_amend_new.txt", "w", encoding="utf8") as g:
    for line in f:
        if line in sen:
            continue

        g.write(line)