from py2neo import Graph, Node, Relationship
import os

CWD = os.getcwd()

ANNOTATION_DIR = CWD + "\\sentences_annotation_auto\\"
ANNOTATION_TARGET = ANNOTATION_DIR + "annotation.txt"
ANNOTATION_INFOBOX = ANNOTATION_DIR + "annotation_infobox.txt"

DB = Graph(
    "bolt://localhost:7687",
    username="neo4j",
    password="admin"
)


def split_line(line):
    line = line.strip()
    spl = line.rsplit(" ", 3)
    return spl[0], spl[1], spl[2], spl[3]


with open(ANNOTATION_TARGET, "r", encoding="utf8") as f:
    lines = f.readlines()

with open(ANNOTATION_INFOBOX, "w", 1, encoding="utf8") as g, \
        open(ANNOTATION_DIR + "annotate_error.txt", "a", 1, encoding="utf8") as e:
    for line in lines:
        print(line)
        if line.startswith("@@"):
            continue
        sentence, entity_a, entity_b, relation = split_line(line)
        try:
            match = DB.data(
                "MATCH p=(a:person {{name:'{}'}})-[r:{}]-(b:person {{name:'{}'}}) WHERE r.infobox>=1 RETURN count(*)".format(
                    entity_a, relation, entity_b))
            if int(match[0]['count(*)']) >= 1:
                g.write(line)
        except:
            e.write(line)
