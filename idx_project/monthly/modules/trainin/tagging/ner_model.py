import tensorflow as tf
from gensim.models import Word2Vec


def load_data():
    sentences = []
    sentence = []
    words = []
    label_line=[]
    labels = []
    with open("../../../data/OrientedTag") as f:
        contents = f.readlines()
        for content in contents:
            c = content.split(" ")
            if len(c) == 2:
                w = c[0]
                l = c[1].replace("\n", "")
                words.append(w)
                # labels.append(l)
                label_line.append(l)
                sentence.append(w)
            else:
                labels.append(label_line)
                sentences.append(sentence)
                sentence = []
                label_line=[]

    return zip(sentences, labels)


def init_session():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess


def sentence_convert(self, sentences):
    idx_sentences = []
    idx_sentence = []
    for sentence in sentences:
        # print(sentences.txt)
        for word in sentence:
            # print(word)
            try:
                idx_sentence.append(self.w2vDic[word])
            except:
                pass
        idx_sentence = []
        idx_sentences.append(idx_sentence)
    return idx_sentences

def _seq_padding(sequences,pad_tok,max_length):
    sequence, sequence_length = [], []
    for s in sequences:
        s = list(s)
        seq_ = s[:max_length] + [pad_tok] * max(max_length - len(s), 0)
        # print(seq_)
        sequence_length += [min(len(sequences), max_length)]
        sequence.append(seq_)
    return sequence,sequence_length

def seq_padding(sequences,pad_tok,option=1):
    """

    :param seq:
    :return: a list of list where each sublist has same length
    """
    sequence = []
    sequence_length = []
    if option==1:
        # padding for list of words
        max_length = max(map(lambda x: len(x), sequences))
        sequence,sequence_length=_seq_padding(sequences,pad_tok,max_length)
        sequence_length=[sequence_length]
    elif option==2:
        # padding for list of chars
        max_length = max(map(lambda x: len(x), sequences))
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sentence_list=[]
        for seq in sequences:
            sentence = []
            for word in seq:
                word=list(word)
                seq_ = word[:max_length_word] + [pad_tok] * max(max_length_word - len(word), 0)
                sentence.append(seq_)

            l=[0]*max_length_word
            for t in range(max_length-len(sentence)):
                sentence.append(l)
            sentence_list.append(sentence)
            sequence_length.append(max_length_word)
        sequence=sentence_list

        #     sp, sl = _seq_padding(seq,pad_tok, max_length_word)
        #     sequence += [sp]
        #     sequence_length += [sl]
        #
        # max_length_sentence = max(map(lambda x: len(x), sequences))
        # sequence_padded, _ = _seq_padding(sequence,
        #                                     [pad_tok] * max_length_word, max_length_sentence)
        # sequence_length, _ = _seq_padding(sequence_length,pad_tok,
        #                                     max_length_sentence)
    return sequence, sequence_length


class NERModel:
    def __init__(self):
        """ initialize"""
        self.ntags = 3  # ('O','Oriented-S','Oriented-C')
        self.tags = {'O': 0, 'Oriented-S': 1, 'Oriented-C': 2}
        # softmax_model hyper parameters
        self.hidden_size_char = 100  # lstm on chars
        self.hidden_size_lstm = 300  # lstm on word embeddings

        # to store the loaded softmax_model
        self.w2vModel = None  # to be initialize in tfAdapt()
        self.w2vDic = {}  # to be initialize  in tfAdapt()
        self.chDic = {}  # to be initialize  in loadChar()
        self.w2v_embeddings = None  # to be adapted in tfAdapt() from gensim to tf
        self.lstm_embeddings = None  # to be initialized  in lstmEmbed()
        self.word_embeddings = None  # to be initialized in wordRepresentation()
        self.context_rep = None  # to be initialized in contextRepresentation()
        self.scores = []  # to be initialized in decoding()

        # for words look up: refer to the word by word_id in the w2vEmbeddings
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # for chars look up: refer to the char by char_id in the lstmEmbeddings
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

    def loadChar(self, filename):
        """

        :param filename: the path to the file which store all characters
        :return: chList--character list, chDic-- character dictionary to store the index
        """
        chList = []
        chDic = {}
        id = 0
        with open(filename) as f:
            content = f.readlines()
            for c in content:
                c = c.replace("\n", "")
                chDic[c] = id
                id += 1
                chList.append(c)
        self.chDic = chDic
        return chList

    def tfAdapt(self):
        """

        :param w2vModel: the word2vec softmax_model( loaded by the build-in Word2Vec.load())
        :return: encapsulate the word2vec softmax_model into the tf.nn.embedding_lookup()
        """
        w2vModel = Word2Vec.load("../../../softmax_model/w2vModel")
        self.w2vModel = w2vModel
        vocabList = w2vModel.wv.index2word  # get the list of vocabulary from the preprocess softmax_model
        embeddings = w2vModel[vocabList]  # 2-dimensional array shaped (315, 100)
        # print(vocabList)
        for i in range(len(vocabList)):
            self.w2vDic[vocabList[i]] = i

        # print(vocabList)
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False, name="L")  # type casting
        w2v_embeddings = tf.nn.embedding_lookup(L, self.word_ids, name="w2v_embeddings")

        self.w2v_embeddings = w2v_embeddings
        """
        # printing out
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            print(sess.run(L))  # the result is the same as print(embedding) except for the type
            print(sess.run(pretrained_embeddings, {self.word_ids: [[0], [1], [2]]}))  # printing 0-th 1-th and
        """

    def lstmEmbed(self):
        """

        :return: lstm-trained vector in character level
        """
        chars = self.loadChar("../../../data/chars")
        nchars = len(chars)
        dim_char = 100  # the same as the preprocess softmax_model

        # 1. get character embeddings
        K = tf.get_variable(dtype=tf.float32, shape=[nchars, dim_char],
                            name="K")  # tf.get_variable can form a matrix randomly
        raw_embeddings = tf.nn.embedding_lookup(K, self.char_ids, name="raw_char_embeddings")

        # 2. put the time dimension on axis=1 for dynamic_rnn
        shape = tf.shape(raw_embeddings, name="shape")
        reshaped_embeddings = tf.reshape(raw_embeddings,
                                         shape=[shape[0] * shape[1], shape[-2], dim_char],
                                         name="reshaped_char_embeddings")
        word_lengths = tf.reshape(self.word_lengths, shape=[shape[0] * shape[1]], name="word_lengths")

        # bi lstm on chars
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_char, state_is_tuple=True, name="cell_fw")  # forwoard
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_char, state_is_tuple=True, name="cell_bw")  # backward

        _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                              cell_bw, reshaped_embeddings,
                                                                              sequence_length=word_lengths,
                                                                              dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1, name="output")  # concatenating fw and bw (len: 100+100=200)
        trained_embeddings = tf.reshape(output, shape=[-1, shape[1], 2 * self.hidden_size_char],
                                        name="trained_embeddings")
        self.lstm_embeddings = trained_embeddings
        """
        # printing out
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # print(sess.run(K))
            chars_id=[[[0]]]  # lookup for the vector for character 0
            word_lengths=[[1]]
            print("raw char_embeddings:")
            print(sess.run(raw_embeddings,
                           {self.char_ids: chars_id}))  # how many "[" are needed depends on self.char_ids
            print("raw embeddings shape: ")
            print(sess.run(shape, {self.char_ids: chars_id}))  # [  1   1   1 100]
            print("\nreshaped char_embeddings:")
            print(sess.run(reshaped_embeddings, {self.char_ids: chars_id}))
            print("\ntrained embeddings:")
            print("output_fw shape:")
            print(str(sess.run(tf.shape(output_fw), {self.char_ids: chars_id,self.word_lengths:word_lengths})))
            print("output_bw shape:)")
            print(str(sess.run(tf.shape(output_bw), {self.char_ids: chars_id, self.word_lengths: word_lengths})))
            print("trained_embeddings shape:")
            print(str(sess.run(tf.shape(trained_embeddings), {self.char_ids: chars_id, self.word_lengths: word_lengths})))
        """

    def indexChar(self, wordindex):
        """
        Converting the index of a word to group of char index
        (input:'散货船'-9 ==> wordindex=9
         output: '散'-234 ,'货'-235,'船'-11 ==>[[[234,235,11]]])

        :param wordindex: index for the preprocess vocabulary list
        :param chDic: dictionary to find out the index of a character in the char list
        :param w2vModel: word2vec softmax_model
        :return: char_ids containing the index of the characters from the word
        """

        # print(w2vModel.wv.index2word[wordindex])
        # print(w2vModel.wv['散货船'])

        word = self.w2vModel.wv.index2word[wordindex]
        char_ids = []
        # print(self.w2vModel.wv.index2word[wordindex])
        for c in word:
            # print(c + ": " + str(chDic[c]))
            char_ids.append(self.chDic[c])
        char_ids = [[char_ids]]
        word_ids = [[wordindex]]
        return char_ids, word_ids

    def wordRepresentation(self):
        """
        the word vector is consisted of 2 parts:
            A=(word2vec vector)        # len(A)=100
            B=(lstm trained vector( concatenating bw and fw))  #len(B)=200
            final vector= A+B     # note that '+' is the concatenating operation rather than numeric calculation
        :return: final vector
        """

        word_embeddings = tf.concat([self.w2v_embeddings, self.lstm_embeddings], axis=-1)

        self.word_embeddings = word_embeddings
        """
        # printing out
        sess=init_session()

        # print(sess.run(w2vEmbeddings, {self.word_ids: [[9]]}))
        # print(sess.run(tf.shape(lstmEmbeddings), {self.char_ids: [[[0]]], self.word_lengths: [[1]]}))

        word_ids = 9  # index of
        char_ids,word_ids = self.indexChar(word_ids)
        print(char_ids)
        print(word_ids)
        print(sess.run(tf.shape(self.word_embeddings),
                       {self.word_ids: [[9]], self.char_ids: [[[68, 60, 0, 52, 5]]], ner.word_lengths: [[1]]}))
         # """

    def contextRepresentation(self):

        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw, self.word_embeddings,
                                                                    sequence_length=self.sequence_lengths,
                                                                    dtype=tf.float32)

        context_rep = tf.concat([output_fw, output_bw], axis=-1)

        self.context_rep = context_rep

        """
        # printing out
        sess=init_session()
        word_ids = 9  # index of
        char_ids, word_ids = self.indexChar(word_ids)
        # print(sess.run(tf.shape(self.word_embeddings),
        #                {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]]}))
        print(sess.run(tf.shape(context_rep),
                       {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]],
                        self.sequence_lengths: [1]}))
        # """

    def decoding(self):
        # tf.get_variable() create variable randomly
        W = tf.get_variable("W", shape=[2 * self.hidden_size_lstm, self.ntags],
                            dtype=tf.float32)

        b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        ntime_steps = tf.shape(self.context_rep)[1]
        context_rep_flat = tf.reshape(self.context_rep, [-1, 2 * self.hidden_size_lstm])  # from [1 1 600] to [1 600]
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, self.ntags])  # scores for 3 tags ('O','Oriented-S','Oriented-C')
        self.scores = scores
        """
        # print out
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            word_ids = 10  # index of
            char_ids, word_ids = self.indexChar(word_ids)
            print(
                sess.run(tf.shape(self.context_rep),
                         {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]],
                          self.sequence_lengths: [1]}))
            print(sess.run(tf.shape(context_rep_flat),
                           {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]],
                            self.sequence_lengths: [1]}))
            print(sess.run(pred,
                           {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]],
                            self.sequence_lengths: [1]}))
            print(sess.run(scores,
                           {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]],
                            self.sequence_lengths: [1]}))
        # """

    def process_word(self):
        """

        :return:
        f("散货船") = ([234, 235, 11], 9)
        = (list of char ids, word id)
        """

        w2vDic = self.w2vDic
        chDic = self.chDic

        def f(word):
            # get ids of char
            char_ids = []
            for char in word:
                if char in chDic.keys():
                    char_ids += [chDic[char]]
            # get id of word
            if word in w2vDic.keys():
                word = w2vDic[word]

            return char_ids, word

        return f

    def create_dev(self):
        sentences = []
        tag_list=[]
        words=[]
        tags = []
        data = load_data()
        for sentence, labels in data:
            for w,l in zip(sentence,labels):
                # print(word+"  "+label)
                f = self.process_word()
                word = f(w)
                label = self.tags[l]
                words += [word]
                tags += [label]
                # print(str(word) + "-" + str(label) )
            sentences.append(words)
            words=[]
            tag_list.append(tags)
            tags=[]
        # try:
        #     print("loading...")
        #     with open("../../../data/dev/data", "r") as file:
        #         contents = file.readlines()
        #         for line in contents:
        #             d = line.split("-")
        #             tup = d[0].split("],")
        #             t1 = [int(i) for i in tup[0].strip("([").replace(" ", "").split(",")]
        #             t2 = int(tup[1].strip(")").strip())
        #             word = (t1, t2)
        #             tag = int(d[1])
        #             words += [word]
        #             tags += [tag]
        # except:
        #     print("processing...")
        #     # with open("../../../data/dev/data", "w") as file:
        #     data = load_data()
        #     for w, l in data:
        #         # print("processing: "+w)
        #         f = self.process_word()
        #         word = f(w)
        #         label = self.tags[l]
        #         words += [word]
        #         tags += [label]
        #         print(str(word) + "-" + str(label) )
        #         # file.write(str(word) + "-" + str(label) + "\n")
        return sentences, tag_list

    def get_feed_dict(self, sentences, labels):
        """

        :param words:
        :param labels:
        :return: a dictionary holding all the needed parameter
        """
        char_ids=[]
        word_ids=[]
        for item in sentences:
            chars, words=zip(*item)
            word_ids.append(list(words))
            char_ids.append(list(chars))

        word_ids, sequence_lengths = seq_padding(word_ids, 0,option=1)  # 1 for words
        # print(word_ids)
        # print(sequence_lengths)
        char_ids, word_lengths = seq_padding(char_ids,0, option=2)  # 2 for chars
        # print(char_ids)
        # print(len(word_lengths))
        # for item in char_ids[0]:
        #     print(len(item))

        # char_ids, word_ids = zip(*words)
        # char_ids, word_lenth = seq_padding(char_ids)  # tensorflow accept list whose sub-lists are the same in length
        """Input to reshape is a tensor with 276 values, but the requested shape has 22632
        feet_dict = {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [word_lengths],
                     self.sequence_lengths: [82]}
        """
        feet_dict = {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [word_lengths*82],
                     self.sequence_lengths: [82]}

        return feet_dict

    def build(self):
        self.tfAdapt()  # word level(preprocess embeddings), init self.w2v_embeddings
        self.lstmEmbed()  # character level(lstm_embeddings), init self.lstm_embeddings
        self.wordRepresentation()
        self.contextRepresentation()
        self.decoding()  # initialize scores

    def train(self):
        self.build()

        words, labels = self.create_dev()

        fd = self.get_feed_dict(words, labels)
        print(fd)
        sess = init_session()
        # print(sess.run(self.word_ids, feed_dict=fd))
        # print(sess.run(self.char_ids, feed_dict=fd))
        # print(sess.run(self.word_embeddings, feed_dict=fd))
        print(sess.run(self.context_rep, feed_dict=fd))
        # log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        #     self.scores, self.labels, self.sequence_lengths)
        #
        # loss = tf.reduce_mean(-log_likelihood)
        #
        # optimizer = tf.train.AdamOptimizer(0.001)
        # train_op = optimizer.minimize(loss)
        #
        # sess.run(train_op,feed_dict=fd)
        """
        # print out
        sess = init_session()
        print(sess.run(trans_params))
        word_ids = 9  # index of
        char_ids, word_ids = self.indexChar(word_ids)
        # print(sess.run(log_likelihood,
        #                {self.word_ids: word_ids, self.char_ids: char_ids, self.word_lengths: [[1]],
        #                 self.sequence_lengths: [1], self.labels:labels}))
        # """


if __name__ == '__main__':
    ner = NERModel()
    ner.train()
