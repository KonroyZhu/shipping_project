import jieba
from gensim.models import Word2Vec
import numpy as np


class Data_util:
    def char2index(self):
        char2index = {}
        with open("../../data/processed/char.txt") as f:
            char_list = f.readlines()
            idx = 1
            for char in char_list:
                char2index[char.replace("\n", "")] = idx
                idx += 1
        return char2index

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
        words=[]
        with open("../../data/processed/data.txt") as f:
            content=f.readlines()
            for line in content:
                arr = line.split("  ")
                if len(arr) > 1 and arr[0] != '':
                    words.append(arr[0])

                elif len(arr) == 1:
                    sentence_list.append(words)
                    words=[]

        return sentence_list

    def tag2index(self):
        return {"Ori-S": 2, "Ori-I": 3, "Ori-E": 4, "O": 1,"A":0}

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
        char2index = self.char2index()
        tag2index = self.tag2index()
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
            if len(arr) > 1 and arr[0]!='':
                x.append(arr[0])
                y.append(arr[1].strip())
            elif len(arr)==1:
                x_len = len(x)
                if x_len < max_length:
                    x += [" "] * (max_length - x_len)
                    y += ["A"] * (max_length - x_len)
                y_t = [tag2index[c] for c in y]
                if option == 1:
                    x_t = [char2index[c] for c in x]
                elif option == 2:
                    x_t = [w2v.wv[c]*20 for c in x]
                x = []
                y = []
                res_sen_list.append(x_t)
                tag_list.append(y_t)
        # """
        return np.array(res_sen_list), np.array(tag_list)

# data_util = Data_util()
# sen_list, tag_list = data_util.load_data(1)
# print(sen_list.shape)


