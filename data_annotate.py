import os

SENTENCE_DIR = "D:\\Projects\\Baike\\relation_sentences"
SENTENCE_ANNO_DIR = "D:\\Projects\\Baike\\sentences_annotation\\"

dirs = os.listdir(SENTENCE_DIR)
flag = True
for file in dirs:
    filename = SENTENCE_DIR + os.sep + file
    basename = os.path.basename(filename)
    print("文件名：" + basename)

    if flag:
        if "乔恩·沃伊特" in basename:
            flag = False
        if flag:
            continue
    with open(filename, 'r', encoding="utf8")as f:

        while True:
            f.seek(0)
            g = open(SENTENCE_ANNO_DIR + basename, "w", encoding="utf8")
            for li in f:
                line = li.strip()
                if len(line) == 0:
                    continue
                print(line + ": ", end="")
                inp = input()
                if inp == "1":
                    g.write(line + "\n")
            g.close()
            print("重做吗: ")
            inp = input()
            if inp != '1':
                break
