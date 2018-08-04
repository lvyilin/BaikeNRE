import sqlite3

conn = sqlite3.connect('baike.db')

c = conn.cursor()

# Create table
c.execute("DROP TABLE relations")
c.execute('''CREATE TABLE relations 
             (name text primary key, relation text, type text)''')

with open("person_relation.txt","r") as f:
    for line in f:
        li = line.split(" ")
        try:
            c.execute("INSERT OR REPLACE INTO relations VALUES ('{:s}','{:s}','{:s}')".format(li[0],li[1],li[2]))
        except sqlite3.IntegrityError as e:
            print(line)


# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
