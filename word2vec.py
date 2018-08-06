from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile
from gensim.test.utils import datapath
from gensim.models import Word2Vec
import os

CWD = os.getcwd()
sentences = LineSentence(datapath(CWD + '\\separated_corpus_patch.txt'))
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, iter=10)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

path = CWD + "\\wordvectors.kv"
model.wv.save(path)
