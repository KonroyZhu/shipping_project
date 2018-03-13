import sklearn_crfsuite
from sklearn_crfsuite import metrics
import jieba.posseg as pseg

from idx_project.idx_project.news_tagging.process.crf.preprocess.data_utils import convert

data_set = convert()
sp = int(len(data_set) * 0.8)
train_data, test_data = data_set[:sp], data_set[sp:]


# '''
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    s = "".join([c[0] for c in sent])
    idx = s.index(word)
    features = {
        'bias': 1.0,
        'word.isdigit()': word.isdigit(),  # whether or not word is a digit
        'postag': postag,  # pos tag of the word
        'iselement': (word in "散货 国际 油 轮船 煤炭 市场 钢材"),  # whether or not the word is from pre-defined-dictionary
        'index': idx  # index of words form the sentence
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:postag': postag1
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:postag': postag1
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

X_test = [sent2features(s) for s in test_data]
y_test = [sent2labels(s) for s in test_data]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')
print(labels)

# the result for all labels
y_pred = crf.predict(X_test)
rate = metrics.flat_f1_score(y_test, y_pred,
                             average='weighted', labels=labels)

for lab, pred in zip(y_test, y_pred):
    print("label: ", lab)
    print("predict:", pred)
print(rate)

# result for each labels
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

sentence = "比起二手船市场，新造船市场明显就稍显平淡了"
sentence = [tuple(c) for c in pseg.cut(sentence)]
print(sentence)
feature = sent2features(sentence)
predict = crf.predict([feature])
for s,l in zip(sentence,predict[0]):
    print(s[0]+" "+l)


# '''
