from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA



# loading our preprocess softmax_model
model=Word2Vec.load("../../softmax_model/w2vModel")


def dimensionReduction(X):
    """

    :param X: high-dimensional vector
    :return: 2-dimensional vector
    """
    # decreasing dimensions
    pca = PCA(n_components=2)  # a 2-dimensional PCA softmax_model
    result = pca.fit_transform(X)  # fitting the 100-dimensional preprocess softmax_model into a 2-dimensional PCA softmax_model
    return result

def plotting(dataList):
    """

    :param dataList: 2-dimensional list
    :return: plotting data in 2 scatter form
    """
    # adapting pyplot to Chinese characters
    pyplot.rcParams[u'font.sans-serif'] = ['simhei']
    pyplot.rcParams['axes.unicode_minus'] = False

    pyplot.scatter(dataList[:, 0], dataList[:, 1])  # create the scatter with data from 'result'
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(dataList[i, 0], dataList[i, 1]))  # adding label for each single point
    pyplot.show()


if __name__ == '__main__':
    X = model[model.wv.vocab]
    result=dimensionReduction(X)
    plotting(result)