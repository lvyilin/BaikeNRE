import os
import re

CORPUS = 'corpus_clean.txt'
NEW_CORPUS = 'corpus_nopunc.txt'
with open(CORPUS, "r", encoding='utf8') as f, open(NEW_CORPUS, 'w', encoding='utf8')as g:
    for line in f:
        content = line.split('\t')
        sentence = content[4]
        sentence = re.sub('[，。；《》？！：、～“”‘’【】（）]', '', sentence)
        d = sentence.split(" ")
        sentence = " ".join(sentence.split())
        content[4] = sentence + "\n"
        g.write("\t".join(content))
