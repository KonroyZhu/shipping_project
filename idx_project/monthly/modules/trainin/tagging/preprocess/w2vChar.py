from gensim.models import Word2Vec

#load and to char
segSentenceList= []
with open("../../../../data/sentences.txt") as f:
    senList=f.readlines()
    for sentence in senList:
        char=[c for c in sentence]
        segSentenceList.append(char)

# print(segSentenceList)
model = Word2Vec(segSentenceList, size=100, window=5, min_count=0)
print(model.wv['B'])
# softmax_model.save("./softmax_model/preprocess")
#
# with open("./data/vocab.txt","w") as f:
#     for v in softmax_model.wv.vocab:
#         print(v)
#         f.write(v+"\n")

