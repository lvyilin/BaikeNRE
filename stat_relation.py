from py2neo import Graph, Node, Relationship

def load_relation():
    d = set()
    with open("person_relation.txt", "r", encoding="utf8") as f:
        for line in f:
            li = line.split(" ")
            # ENTITY_MAP.add(line.split(" ")[0])
            # d[li[0]] = (li[1], li[2].rstrip())
            d.add(li[1])
    return d

RELATION_SET = load_relation()

m_Graph_DB = Graph(
    "bolt://localhost:7687",
    username="neo4j",
    password="admin"
)

with open("relation_body_stat.txt","w") as f:
    for rel in RELATION_SET:
        data = m_Graph_DB.data("MATCH ()-[r:{0}]->() WHERE r.body>=1 RETURN count(*)".format(rel))
        f.write("{0} {1}\n".format(rel,data[0]['count(*)']))