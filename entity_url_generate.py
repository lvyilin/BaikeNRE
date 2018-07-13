prefix = "http://baike.baidu.com/item/"
with open("entityname_format.txt", "r", encoding='utf-8') as f:
    with open("entity_url.txt", "w", encoding='utf-8') as g:
        for line in f:
            g.write("%s" % (prefix + line))
