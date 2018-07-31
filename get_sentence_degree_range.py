import sqlite3
import os

CWD = os.getcwd()

DATA = []


def split_sentence(line):
    spl = str(line).split(" ", 2)
    return spl[0], spl[1], spl[2]


def save_to_file():
    with open("co_occur_stats_filtered.txt", "w", encoding="utf8") as f:
        for item in DATA:
            f.write("{} {} {}\n".format(item[0], item[1], item[2]))


def save_to_sqlite():
    conn = sqlite3.connect('baike.sqlite')
    c = conn.cursor()
    for item in DATA:
        item[0] = str(item[0]).replace("'", "''")
        item[1] = str(item[1]).replace("'", "''")
        item[2] = str(item[2]).replace("'", "''")
        # item[2] = str(item[2]).replace('"','""')
        sql = "insert into Data(entity_a,entity_b,sentence,relation) VALUES('{}','{}','{}',0)".format(item[0], item[1],
                                                                                                      item[2])
        print(sql)
        c.execute(sql)

    conn.commit()
    conn.close()


def read_data():
    entity_set = set()
    with open("co_occur_stats_lite.txt", "r", encoding="utf8") as g:
        for line in g:
            spl = line.split("\t")
            entity_set.add(spl[0])

    with open("entity_sentences_lite.txt", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            entity_a, entity_b, sentence = split_sentence(line)
            if entity_a in entity_set or entity_b in entity_set:
                DATA.append([entity_a, entity_b, sentence])


def main():
    read_data()
    # save_to_file()
    save_to_sqlite()


if __name__ == '__main__':
    main()
