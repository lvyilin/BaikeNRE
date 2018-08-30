# 修正fail_sentence到语料
import os

CWD = os.getcwd()
AMEND_FILE = CWD + "\\fail_sentence_cnssnn_amended.txt"

ID_CORPUS = CWD + "\\separated_corpus_with_label_patch_id.txt"
CORPUS_AMEND = CWD + "\\separated_corpus_with_label_patch_amend.txt"

with open(ID_CORPUS, "r", encoding="utf8") as f, open(AMEND_FILE, "r", encoding="utf8") as g, open(CORPUS_AMEND, "w",
                                                                                                   encoding="utf8") as h:
    amend_id = []
    amend_content = []
    for line in g:
        spl = line.split(" ", maxsplit=1)
        idx = int(spl[0])
        content = spl[1]
        amend_id.append(idx)
        amend_content.append(content)

    output_content = []
    for line in f:
        spl = line.split(" ", maxsplit=1)
        idx = int(spl[0])
        content = spl[1]
        if idx in amend_id:
            output_content.append(amend_content[amend_id.index(idx)])
        else:
            output_content.append(content)

    h.write("".join(output_content))
