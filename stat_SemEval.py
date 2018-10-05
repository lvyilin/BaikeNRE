import os

CWD = os.getcwd()
CORPUS_TRAIN = os.path.join(CWD, "corpus_train_SemEval.txt")
CORPUS_TEST = os.path.join(CWD, "corpus_test_SemEval.txt")
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
counts = [0 for i in range(19)]
entity_dict = {}

for corpus in (CORPUS_TRAIN, CORPUS_TEST):
    with open(corpus, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split("\t")
            en1 = content[1]
            en2 = content[2]
            relation = int(content[5])
            counts[relation] += 1
            for en in (en1, en2):
                if en not in entity_dict:
                    entity_dict[en] = set()
            if en2 not in entity_dict[en1]:
                entity_dict[en1].add(en2)
            if en1 not in entity_dict[en2]:
                entity_dict[en2].add(en1)

print("样本数: %d" % sum(counts))
for rel, cnt in zip(RELATION, counts):
    print("%s: %d" % (rel, cnt))
print("实体数: %d" % len(entity_dict))
degree = [len(s) for s in entity_dict.values()]
print("平均度数: %f" % (sum(degree) / len(degree)))
print("最大度数: %d" % max(degree))
print("最小度数: %d" % min(degree))

edge_set = set()
for k, v in entity_dict.items():
    for en2 in v:
        if (k, en2) not in edge_set and (en2, k) not in edge_set:
            edge_set.add((k, en2))
print("边数: %d" % len(edge_set))
