from idx_project.monthly.modules.trainin.loader import Loader

loader=Loader()
sentenceSeg=loader.load("sentenceSeg")
tags=loader.load("tag")

tagList=[]


for i in range(len(sentenceSeg)):
    for word in sentenceSeg[i].split(" "):
        if word != "\n" and word !="":
            tag=tags[i].replace("1","")
            # print(tag)
            if word in tag:
                # print(tag+" "+word+"  oriented")
                B=''.join([w[0] for w in tag.split(" ") if w !=""])
                if word[0] in B:
                    # print(word+" Oriented-S")
                    tagList.append(word+" Oriented-S")
                else:
                    # print(word+" Oriented-C")
                    tagList.append(word + " Oriented-C")
            else:
                # print(word+"  O")
                tagList.append(word + " O")
    tagList.append("")


with open('../../data/OrientedTag','w') as f:
    for item in tagList:
        print(item)
        f.write(item+"\n")
