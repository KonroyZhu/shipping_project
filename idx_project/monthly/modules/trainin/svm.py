import  numpy as np
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split

from idx_project.monthly.modules.trainin.loader import Loader


def getVecList():
    '''
        1. load sentences from data file
        2. load the w2vModel from softmax_model file
        3. traverse the sentences.txt from sentences.txt list:
            adding every word vectors from a sentences.txt together and return the average
        4. store them inside the variable called vecList
        :return
            sentences represented by vectors featured by word2vec
    '''
    loader=Loader()
    sentenceList=loader.loadSentence()
    model=Word2Vec.load("../../softmax_model/w2vModel")

    print(model.wv.vocab)
    size=len(model.wv['\n'])
    vecList=[]
    for sentence in sentenceList:
        # vector=np.zeros(size).reshape((1, size))
        vector= np.zeros(size)
        count=0.
        for words in sentence:
            try:
                vector+=model.wv[words]
                count=count+1
            except:
                # without taking words whose frequency are lower than min_count into account
                pass
        vector=(vector/count)*1000
        vecList.append(vector)
    return vecList

def format():
    loader=Loader()
    tagList=loader.loadTag()
    vecList=getVecList()
    tagList=[len(i) for i in tagList]
    X=np.array(vecList)
    y=np.array(tagList)
    return X,y

X,y=format()
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2)


from sklearn.svm import SVC
model=SVC()
model.fit(trainX,trainy)



prediction=model.predict([x for x in testX])# the right input for 'predict' is a 2-dimension list
total=len(prediction)
hits=0
for i in range(total):
    print(str(prediction[i])+" "+str(testy[i]))
    if prediction[i]==testy[i]:
        hits=hits+1
print(hits/total)
