from gensim.models import Word2Vec
import numpy as np
import pickle


class Data_util:
    def __init__(self):
        self.max_length=194
        self.w2v_times=10

    def char2index(self):
        index2char = {}
        char2index = {}
        with open("../../data/processed/char.txt") as f:
            char_list = f.readlines()
            idx = 1
            for char in char_list:
                ch = char.replace("\n", "")
                index2char[idx] = ch
                char2index[ch] = idx
                idx += 1
        return char2index, index2char

    def load_w2v(self):
        model = Word2Vec.load("../../data/processed/w2v_char")
        return model

    def load_sentence(self):
        sentence_list = []
        # with open("../../data/sentences.txt") as f:
        #     content = f.readlines()
        #     for line in content:
        #         char_list = [c for c in line]
        #         sentence_list.append(char_list)
        words = []
        with open("../../data/processed/data.txt") as f:
            content = f.readlines()
            for line in content:
                arr = line.split("  ")
                if len(arr) > 1 and arr[0] != '':
                    words.append(arr[0])

                elif len(arr) == 1:
                    sentence_list.append(words)
                    words = []

        return sentence_list

    def tag2index(self):
        tag2index = {"Ori-S": 2, "Ori-I": 3, "Ori-E": 4, "O": 1, "A": 0}
        index2tag = {2: "Ori-S", 3: "Ori-I", 4: "Ori-E", 1: "O", 0: "A"}
        return tag2index, index2tag

    def load_data(self, option=1):
        """

        :param option:
            1: load with words' id in sentence_list
            2: load with w2v words' vector in sentence_list
        :return:
            option 1:
            sentence represented by list of words' id in sentence_list
            labels for words in each sentences
            option 2:
            sentence represented by list of words' vector in sentence_list
            labels for words in each sentences

        """
        char2index, index2char = self.char2index()
        tag2index, index2tag = self.tag2index()
        sentence_list = self.load_sentence()
        w2v = Word2Vec.load("../../data/processed/w2v_char")
        f = open("../../data/processed/data.txt")
        content = f.readlines()
        max_length = max(map(lambda k: len(k), sentence_list))

        res_sen_list = []
        tag_list = []
        x = []
        y = []
        # """
        for line in content:
            x_t = []
            y_t = []
            line = line.replace("\n", "")
            arr = line.split("  ")
            if len(arr) > 1 and arr[0] != '':
                x.append(arr[0])
                y.append(arr[1].strip())
            elif len(arr) == 1:
                x_len = len(x)
                if x_len < max_length:
                    x += [" "] * (max_length - x_len)
                    y += ["A"] * (max_length - x_len)
                y_t = [tag2index[c] for c in y]
                if option == 1:
                    x_t = [char2index[c] for c in x]
                elif option == 2:
                    x_t = [w2v.wv[c] * self.w2v_times for c in x]
                x = []
                y = []
                res_sen_list.append(x_t)
                tag_list.append(y_t)
        # """
        return np.array(res_sen_list), np.array(tag_list), char2index, tag2index, index2char, index2tag

    def convert(self,sentence,option=1):
        """

        :param sentence: string of sentence
        :param option:
                        1: convert to list of id
                        2: convert to list of w2v vector
        :return:
        """
        char_list=[c for c in sentence]
        if len(char_list)<self.max_length:
            char_list+=(self.max_length-len(char_list))*[" "]
        idx_list=[]
        if option==1:
            char2index, index2char = self.char2index()
            idx_list+=[char2index[c] for c in char_list]
        elif option==2:
            w2v=self.load_w2v()
            idx_list+=[w2v.wv[c] * self.w2v_times for c in char_list]
        return idx_list




if __name__ == '__main__':
    data_util = Data_util()
    sentence="国际油轮船价普跌 国际散货船价普涨"
    sentence=data_util.convert(sentence,1)
    print(sentence)
    """
    sen_list, tag_list, char2index, tag2index, index2char, index2tag = data_util.load_data(1)
    print('Starting pickle to file')
    with open("../../data/processed/data.pkl", 'wb') as f:
        # pickle.dump(sen_list, f)
        # pickle.dump(tag_list, f),
        # pickle.dump(char2index, f)
        # pickle.dump(tag2index, f)
        # pickle.dump(index2char, f)
        # pickle.dump(index2tag, f)
        pickle.dump((sen_list, tag_list, char2index, tag2index, index2char, index2tag),f)
    print("Pickle finished")
    """