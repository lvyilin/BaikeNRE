import os, re
from nltk.stem import PorterStemmer

CWD = os.getcwd()

TRAIN_FILE = os.path.join(CWD, "dataset", "SemEval2010_task8_all_data", "SemEval2010_task8_training", "TRAIN_FILE.TXT")
TEST_FILE = os.path.join(CWD, "dataset", "SemEval2010_task8_all_data", "SemEval2010_task8_testing_keys",
                         "TEST_FILE_FULL.TXT")

CORPUS_TRAIN = os.path.join(CWD, "corpus_train_SemEval.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test_SemEval.txt")
CORPUS_ALL = os.path.join(CWD, "corpus_SemEval.txt")

RE_EN1 = re.compile(r"<e1>.*</e1>")
RE_EN2 = re.compile(r"<e2>.*</e2>")
ps = PorterStemmer()

RELATION = ["Cause-Effect(e1,e2)",
            "Cause-Effect(e2,e1)",
            "Instrument-Agency(e1,e2)",
            "Instrument-Agency(e2,e1)",
            "Product-Producer(e1,e2)",
            "Product-Producer(e2,e1)",
            "Content-Container(e1,e2)",
            "Content-Container(e2,e1)",
            "Entity-Origin(e1,e2)",
            "Entity-Origin(e2,e1)",
            "Entity-Destination(e1,e2)",
            "Entity-Destination(e2,e1)",
            "Component-Whole(e1,e2)",
            "Component-Whole(e2,e1)",
            "Member-Collection(e1,e2)",
            "Member-Collection(e2,e1)",
            "Message-Topic(e1,e2)",
            "Message-Topic(e2,e1)",
            "Other"]


def getRelationId(str):
    return RELATION.index(str)


def getEntityPos(sentence: str, entity: str) -> int:
    idx = sentence.find(entity)
    return sentence.count(" ", 0, idx)


def getStem(w):
    try:
        word = w[:-1] if w[-1] is 's' else w
        return word
    except IndexError:
        return ""


sentence_list = []
for file, corpus in ((TRAIN_FILE, CORPUS_TRAIN), (TEST_FILE, CORPUS_TEST)):
    with open(file, "r", encoding="utf8") as fp, open(corpus, "w", encoding="utf8") as gp:
        while True:
            line1 = fp.readline().strip()
            if line1 is "":
                break
            content = line1.split("\t")
            id = int(content[0])
            sentence = content[1]
            line2 = fp.readline().strip()
            relation = getRelationId(line2)
            fp.readline()  # comment
            fp.readline()  # empty line

            punctuation = r"""!"#$%&'()*+,.:;=?@[\]^`{|}~"""
            sentence = sentence.translate(str.maketrans('', '', punctuation))

            en1 = RE_EN1.findall(sentence)[0]
            en2 = RE_EN2.findall(sentence)[0]
            # 如果实体包含多个词，用_合并
            en1_spl = en1.split(" ")
            if len(en1_spl) > 1:
                en1_new = "_".join(en1_spl)
                sentence = sentence.replace(en1, en1_new)
                en1 = en1_new
            en2_spl = en2.split(" ")
            if len(en2_spl) > 1:
                en2_new = "_".join(en2_spl)
                sentence = sentence.replace(en2, en2_new)
                en2 = en2_new

            en1_pos = getEntityPos(sentence, en1)
            en2_pos = getEntityPos(sentence, en2)

            en1 = en1.replace("<e1>", "")
            en1 = en1.replace("</e1>", "")
            en2 = en2.replace("<e2>", "")
            en2 = en2.replace("</e2>", "")
            for old in ("<e1>", "</e1>", "<e2>", "</e2>"):
                sentence = sentence.replace(old, "")

            # en1 = getStem(en1)
            # en2 = getStem(en2)
            en1 = ps.stem(en1.lower())
            en2 = ps.stem(en2.lower())
            sentence_stem = [ps.stem(w.lower()) for w in sentence.split(" ")]
            # sentence_stem = [getStem(w) for w in sentence.split(" ")]
            sentence = " ".join(sentence_stem)
            sentence_list.append(sentence)
            gp.write("%d\t%s\t%s\t%d\t%d\t%d\t%s\n" % (id, en1, en2, en1_pos, en2_pos, relation, sentence))

with open(CORPUS_ALL, "w", encoding="utf8")as fp:
    fp.write("\n".join(sentence_list))
