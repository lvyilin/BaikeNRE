# 将句子分词结果保存
import json
import os
import pyltp

CWD = os.getcwd()
CORPUS = os.path.join(CWD, "sentence_corpus.txt")
CORPUS_LABELED = os.path.join(CWD, "sentence_with_label_corpus.txt")

SEP_CORPUS = os.path.join(CWD, "separated_corpus.txt")
SEP_CORPUS_LABELED = os.path.join(CWD, "separated_corpus_with_label.txt")

LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")


def sentence():
    with open(CORPUS, "r", encoding="utf8") as f, open(SEP_CORPUS, "w", encoding="utf8") as g:
        # i = 0
        for line in f:
            s = []
            words = segmentor.segment(line)
            for w in words:
                s.append(str(w).strip())
                s.append(" ")
            s[-1] = "\n"
            g.write("".join(s))


def sentence_with_label():
    max_sep_word_len = 0
    with open(CORPUS_LABELED, "r", encoding="utf8") as f, open(SEP_CORPUS_LABELED, "w", encoding="utf8") as g:
        for line in f:
            spl = line.split("###")
            s = []
            words = segmentor.segment(spl[0])
            for w in words:
                if len(words) > max_sep_word_len:
                    max_sep_word_len = len(words)
                s.append(str(w).strip())
                s.append(" ")
            s.append(spl[1])
            g.write("".join(s))
    print("max sep word length: %d" % max_sep_word_len)


sentence_with_label()
