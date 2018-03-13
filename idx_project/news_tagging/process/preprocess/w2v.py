from gensim.models import Word2Vec


def load_char():
    sentence_list = []
    with open("../../data/sentences.txt") as f:
        content = f.readlines()
        for line in content:
            char_list = [c for c in line]
            sentence_list.append(char_list)
    return sentence_list


def train_w2v():
    sentence_list = load_char()
    model = Word2Vec(sentence_list, size=100, window=5, min_count=0)
    return model


def save_vocab(vocab_list):
    vocab_list = model.wv.index2word
    with open("../../data/processed/char.txt", "w") as f:
        for v in vocab_list:
            f.write(v + "\n")


model = train_w2v()
vocab_list = model.wv.index2word
save_vocab(vocab_list)
model.save("../../data/processed/w2v_char")
# print(load_char())