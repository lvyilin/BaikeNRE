# 将提取好的包含主体和其他实体的句子文件整合为一个文件
import os

DIR = "D:\\Projects\\Baike\\entity_sentences\\"
FILE = "D:\\Projects\\Baike\\entity_sentences.txt"

files = os.listdir(DIR)
with open(FILE, "w", encoding="utf8") as f:
    for filename in files:
        with open(DIR + filename, "r", encoding="utf8") as g:
            for line in g:
                if line.strip() == "":
                    continue
                f.write(line)
