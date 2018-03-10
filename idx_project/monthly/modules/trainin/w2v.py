from gensim.models import Word2Vec

from idx_project.monthly.modules.trainin.loader import Loader


def w2vTrain(sentenceList):
    '''
        :argument
            the first argument reformedList is a list of list containing words in a sentences.txt:
                here is an example of the format:
                    [['本月', '国际', '二手', '油轮'],[ '各个', '船型', '的', '价格']]
        :parameter
            1. size( the size of the NN layers):
                size=100 means the softmax_model represent a word with a vector of 100 dimensions
                the top most frequently appeared words was selected to become an element in the word-bag
            2. window( the amount of words to be considered during the process of training):
            3. min_count( the lowest time that a word should appear in the training data):
                if the frequency of a certain word is lower than min_count, it would be omitted
    '''
    # softmax_model = Word2Vec(sentenceList, size=100, window=5,min_count=3)
    model = Word2Vec(sentenceList, size=100, window=5,min_count=0)
    print(model.wv.vocab)
    # modelPath="../../softmax_model/w2vModel"
    # softmax_model.save(modelPath)
    # print("softmax_model saved to "+modelPath)


if __name__ == '__main__':
    loader = Loader()
    sentenceList = loader.loadSentence()
    w2vTrain(sentenceList)