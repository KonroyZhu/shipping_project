import jieba
from gensim.models import Word2Vec

def save_vocab(vocab_list):
    vocab_list = model.wv.index2word
    with open("../../data/processed/word.txt", "w") as f:
        for v in vocab_list:
            f.write(v + "\n")

sentence_list=open("../../data/sentences.txt").readlines()
sentence_seg_list=[]
for sentence in sentence_list:
    sentence_seg_list.append([c for c in jieba.cut(sentence,cut_all=False)])

model = Word2Vec(sentence_seg_list, size=100, window=5, min_count=0)
model.save("../../data/processed/w2v_word")
vocab_list = model.wv.index2word
save_vocab(vocab_list)

