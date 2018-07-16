# 将人名生成LTP的外部词典
import json
import os

DIR = "D:\\Projects\\Baike\\parse_data\\"

files = os.listdir(DIR)

with open("entity_dict.txt", "w", encoding="utf8") as f:
    for filename in files:
        with open(DIR + filename, "r", encoding="utf8") as g:
            jsond = json.load(g)
            name = jsond['name']
            f.write(name + "\n")
