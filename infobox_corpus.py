import json
import os

PARSE_DATA_DIR = "D:\\Projects\\Baike\\parse_data"

infobox_dict = {}
for filename in os.listdir(PARSE_DATA_DIR):
    with open(os.path.join(PARSE_DATA_DIR, filename), 'r', encoding='utf8') as fp:
        jsond = json.load(fp)
        infobox_dict[jsond['name']] = jsond['infobox']

with open("corpus_infobox.json", "w", encoding="utf8") as fp:
    json.dump(infobox_dict, fp, ensure_ascii=False)
