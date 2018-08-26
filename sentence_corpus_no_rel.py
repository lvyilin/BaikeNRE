import os
import sqlite3

CWD = os.getcwd()

SENTENCE_LABEL_CORPUS = os.path.join(CWD, "sentence_with_label_corpus_no_rel.txt")
sentence_set = set()
NO_REL_SENTENCE_NUM = 5000


def sentence_with_label():
    conn = sqlite3.connect('baike.db')
    c = conn.cursor()
    c.execute("select count(*) from Data where relation=-1")
    tb1_count = int(c.fetchone()[0])
    c.execute("select count(*) from Data3 where relation=-1")
    tb2_count = int(c.fetchone()[0])
    limit2 = int(NO_REL_SENTENCE_NUM * (tb2_count / (tb1_count + tb2_count)))
    limit1 = NO_REL_SENTENCE_NUM - limit2

    c.execute("select entity_a,entity_b, sentence,relation from Data where relation=-1 order by random() limit ?",
              (limit1,))
    for row in c:
        sentence_set.add(
            row[0].replace(" ", "") + "###" + row[1].replace(" ", "") + "###" + row[2] + "###" + str(row[3]))

    c.execute("select entity_a,entity_b, sentence,relation from Data3 where relation=-1 order by random() limit ?",
              (limit2,))
    for row in c:
        sentence_set.add(
            row[0].replace(" ", "") + "###" + row[1].replace(" ", "") + "###" + row[2] + "###" + str(row[3]))
    with open(SENTENCE_LABEL_CORPUS, "w", encoding="utf8") as g:
        for s in sentence_set:
            g.write(s + "\n")


# sentence()
sentence_with_label()
