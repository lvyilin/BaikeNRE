import os
import sqlite3

CWD = os.getcwd()
SENTENCE_FILE = os.path.join(CWD, "entity_sentences_lite.txt")
SENTENCE_CORPUS = os.path.join(CWD, "sentence_corpus.txt")

SENTENCE_LABEL_CORPUS = os.path.join(CWD, "sentence_with_label_corpus.txt")
sentence_set = set()


def sentence():
    with open(SENTENCE_FILE, "r", encoding="utf8") as f:
        for line in f:
            spl = line.split(" ", maxsplit=2)
            sentence = spl[2]
            sentence_set.add(sentence)

    with open(SENTENCE_CORPUS, "w", encoding="utf8") as g:
        for s in sentence_set:
            g.write(s)


def sentence_with_label():
    conn = sqlite3.connect('baike.db')
    c = conn.cursor()
    c.execute("select entity_a,entity_b, sentence,relation from Data where relation!=0 and relation!=-1")
    for row in c:
        sentence_set.add(row[0] + "###" + row[1] + "###" + row[2] + "###" + str(row[3]))

    c.execute("select entity_a,entity_b, sentence,relation from Data3 where relation!=0 and relation!=-1")
    for row in c:
        sentence_set.add(row[0] + "###" + row[1] + "###" + row[2] + "###" + str(row[3]))
    with open(SENTENCE_LABEL_CORPUS, "w", encoding="utf8") as g:
        for s in sentence_set:
            g.write(s + "\n")


# sentence()
sentence_with_label()
