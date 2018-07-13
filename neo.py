from py2neo import Graph, Node, Relationship


# 此函数每次运行只执行一次
def delete_graph(DB, node = None, relation = None, all = 0):
    if node != None:
        if DB.exists(node):
            DB.delete(node)
    if relation != None:
        if DB.exists(relation):
            DB.delete(relation)
    if all == 1:
        DB.delete_all()

def build_Node(node_name, entity_type="person"):
    n = Node(entity_type, name=node_name)
    return n


def build_Relation(nodeA, nodeB, relation_type, location):
    r1 = Relationship(nodeA, relation_type, nodeB)
    r1[location] = 1
    r1["count"] = 1
    return r1



def build_N_R(m_Graph, nodeA_name, nodeB_name, relation_type, location, entityA_type="person", entityB_type="person"):
    n1 = build_Node(nodeA_name, entityA_type)
    n2 = build_Node(nodeB_name, entityB_type)
    r = build_Relation(n1, n2, relation_type, location)
    if not m_Graph.exists(n1):
        m_Graph.create(n1)

    if not m_Graph.exists(n2):
        m_Graph.create(n2)

    if not m_Graph.exists(r):
        m_Graph.create(r)
    else:
        r["count"] += 1
        r[location] += 1
def main():
    m_Graph_DB = Graph(
        "bolt://localhost:7687",
        username="neo4j",
        password="admin"
    )
    delete_graph(m_Graph_DB,all=1)
    return
    build_N_R(m_Graph_DB, "A", "B" , "friend", "infobox")
    build_N_R(m_Graph_DB, "A", "B" , "friend", "infobox")

if __name__=="__main__":
    main()