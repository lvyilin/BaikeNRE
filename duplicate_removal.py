import string
import os
import json
PARSE_DATA = "D:\\Projects\\Baike\\parse_data\\"
def main():
    person_map = {}
    for filename in os.listdir(PARSE_DATA):
        name = filename.rstrip(".json")
        name = name.rstrip(string.digits)
        info = os.stat(PARSE_DATA+ filename)
        if name not in person_map:
            person_map[name] = [0,'']
        if info.st_size>person_map[name][0]:
            person_map[name][0] = info.st_size
            person_map[name][1] = filename
    # with open('biggest.json','w',encoding='utf-8') as f:
    #     f.write(json.dumps(person_map,ensure_ascii=False))
    person_set = set()
    for k,v in person_map.items():
        person_set.add(v[1])
    for filename in os.listdir(PARSE_DATA):
        if filename not in person_set:
            os.remove(PARSE_DATA+filename)

if __name__ == "__main__":
    main()