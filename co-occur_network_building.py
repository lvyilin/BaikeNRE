from py2neo import Graph, Node, Relationship
import os

CWD = os.getcwd()

SENTENCE_FILE = os.path.join(CWD, "entity_sentences_lite.txt")
DB = Graph(
    "bolt://localhost:7687",
    username="neo4j",
    password="admin"
)


def build_Node(node_name, entity_type="person"):
    n = Node(entity_type, name=node_name)
    return n


def build_Relation(nodeA, nodeB, relation_type):
    r1 = Relationship(nodeA, relation_type, nodeB)
    r1['weight'] = 1
    return r1


def build_N_R(nodeA_name, nodeB_name, relation_type, entityA_type="person", entityB_type="person"):
    n1 = build_Node(nodeA_name, entityA_type)
    DB.merge(n1, entityA_type, 'name')
    n2 = build_Node(nodeB_name, entityB_type)
    DB.merge(n2, entityB_type, 'name')
    r = DB.match_one(n1, relation_type, n2)
    if r is None:
        r = build_Relation(n1, n2, relation_type)
        DB.merge(r)
    else:
        # if r['weight'] is None:
        #     r['weight'] = 1
        # else:
        r['weight'] += 1
        r.push()


def split_sentence(line):
    spl = str(line).split(" ", 2)
    return spl[0], spl[1], spl[2]


def main():
    with open(SENTENCE_FILE, "r", encoding="utf8")as fp:
        for line in fp:
            print(line)
            entity_a, entity_b, sentence = split_sentence(line.strip())
            build_N_R(entity_a, entity_b, "occur")


if __name__ == '__main__':
    main()
