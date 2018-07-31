from py2neo import Graph, Node, Relationship
import os

CWD = os.getcwd()

DB = Graph(
    "bolt://localhost:7687",
    username="neo4j",
    password="admin"
)


def main():
    res = DB.data("MATCH (a)-[r]-() RETURN a.name, sum(r.weight) as sz ORDER BY sz DESC ")
    with open("co_occur_stats_lite.txt", "w", encoding="utf8") as g:
        for item in res:
            if item['sz'] >= 10 and item['sz'] <= 50:
                g.write("{}\t{}\n".format(item['a.name'], item['sz']))


if __name__ == '__main__':
    main()
