sentences = []
labels = []
with open("../../../../data/sentences.txt") as sf:
    sentences = sf.readlines()
with open("../../../../data/tag") as lf:
    labels = lf.readlines()

converted = []

for sentence, label in zip(sentences, labels):
    if label == '\n':
        # print("no orientation")
        for ch in sentence:
            print(ch + "  O")
            converted.append(ch + "  O")  #
    else:
        oriented = str(label).replace("\n", "").replace("-", "").replace("1", "").replace(" ", "").split("#")
        for ori in oriented:
            if ori != "":
                sen_seg = str(sentence).split(ori)
                for c in sen_seg[0]:
                    print(c + "  O")
                    converted.append(c + "  O")  #
                    try:
                        sentence = sen_seg[1]
                    except:
                        pass
                print(ori[0] + "  Ori-S")
                converted.append(ori[0] + "  Ori-S")  #
                for c in ori[1:-1]:
                    print(c + "  Ori-I")
                    converted.append(c + "  Ori-I")  #
                print(ori[-1] + "  Ori-E")
                converted.append(ori[-1] + "  Ori-E")  #
        for c in sentence:
            print(c + "  O")
            converted.append(c + "  O")  #

with open("./data/data.txt","w") as wf:
    for item in converted:
        wf.write(str(item)+"\n")
    wf.write("\n")

