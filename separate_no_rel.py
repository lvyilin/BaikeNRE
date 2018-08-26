# 将句子分词结果保存
import os
import pyltp

CWD = os.getcwd()
CORPUS = os.path.join(CWD, "sentence_corpus.txt")
CORPUS_LABELED = os.path.join(CWD, "sentence_with_label_corpus_no_rel.txt")
SEP_CORPUS = os.path.join(CWD, "separated_corpus.txt")
SEP_CORPUS_LABELED = os.path.join(CWD, "separated_corpus_with_label_no_rel.txt")
FIXED_WORD_LENGTH = 60

LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(cws_model_path, "entity_dict.txt")


def sentence_with_label():
    with open(CORPUS_LABELED, "r", encoding="utf8") as f, open(SEP_CORPUS_LABELED, "w", encoding="utf8") as g:
        i = 0
        for line in f:
            spl = line.split("###")
            t = [spl[0], spl[1], spl[3].strip()]
            spl[2] = spl[2].replace("", " ")
            words = segmentor.segment(spl[2])
            if len(words) > FIXED_WORD_LENGTH:
                continue
            for w in words:
                t.append(str(w).strip())
            i += 1
            g.write("%s\n" % (" ".join(t)))
            if i >= 4000:
                break


# sentence()
sentence_with_label()
