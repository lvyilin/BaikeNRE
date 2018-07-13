import re
from pymongo import MongoClient

regx1 = re.compile(".*人物.*")
regx2 = re.compile(".+")
client = MongoClient()
db = client['test']
collection = db['test1']
# print(collection.find_one({"categories":regx1,"internal_links":regx2}))
cursor = collection.find({"categories": regx1, "internal_links": regx2})

with open("entityname_format.txt", "w", encoding='utf-8') as f:
    for row in cursor:
        name = row['entityname']
        end = name.find('[')
        if end != -1:
            name = name[0:end]
        f.write("%s\n" % (name))
        f.write("%s\n" % (row['entityname']))
