from py2neo import Graph, Node, Relationship


# 此函数每次运行只执行一次
def delete_graph(DB, node = None, relation = None, all =0):
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
    return r1

def add_type_unique(db, type):
    db.run("CREATE CONSTRAINT ON (ea:" + type + ")ASSERT ea.name IS UNIQUE")

def build_N_R(m_Graph, nodeA_name, nodeB_name, relation_type, location, entityA_type="person", entityB_type="person"):
    n1 = build_Node(nodeA_name,entityA_type)
    m_Graph.merge(n1,entityA_type,'name')
    n2 = build_Node(nodeB_name,entityB_type)
    m_Graph.merge(n2,entityB_type,'name')
    r = m_Graph.match_one(n1,relation_type,n2)
    if r == None:
        r = build_Relation(n1, n2, relation_type, location)
        m_Graph.merge(r)
    else:
        if r[location] == None:
            r[location] = 1
        else:
            r[location] += 1
        r.push()

def initDB():
    m_Graph_DB = Graph(
        "bolt://localhost:7687",
        username="neo4j",
        password="admin"
    )
    return m_Graph_DB

def main():
    m_Graph_DB = Graph(
        "bolt://localhost:7687",
        username="neo4j",
        password="admin"
    )
    # add_type_unique(m_Graph_DB,"person")
    delete_graph(m_Graph_DB,all=1)
    return
    build_N_R(m_Graph_DB, "毛泽东", "杨开慧" , "朋友", "infobox")
    build_N_R(m_Graph_DB, "杨开慧", "毛泽东" , "朋友", "infobox")
    build_N_R(m_Graph_DB, "毛泽东", "周恩来" , "同事", "infobox")

if __name__=="__main__":
    main()