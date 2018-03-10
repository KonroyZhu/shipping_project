

chSet=([''])
with open("../../data/sentences.txt") as fr:
    content=fr.readlines()
    for sentence in content:
        for c in sentence:
            if c not in chSet and c !="" and c !="\n":
                chSet.append(c)
fr.close()
print(chSet)
with open('../../data/chars','w') as fw:
    for c in chSet:
        fw.write(c+"\n")
fw.close()