from gensim.scripts.glove2word2vec import glove2word2vec
import os

CWD = os.getcwd()
glove2word2vec(CWD + "\\glove\\vectors.txt", CWD + "\\glove\\word2vec.txt")
