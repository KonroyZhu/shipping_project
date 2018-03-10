import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import logging
import os
from tensorflow.python.framework import ops
from idx_project.monthly.modules.trainin.tagging.data_utils import minibatches, pad_sequences, get_chunks, load_vocab, \
    get_processing_word, CoNLLDataset


class NERModel:
    def __init__(self, embeddings, ntags, logger=None):
        self.embeddings = embeddings
        self.ntags = ntags

        # configuration
        self.hidden_size = 300
        self.crf = True
        self.graph_path = "./graph/"
        self.model_output = "./softmax_model/"
        # self.lr=0.001  # conflictS with the placeholder from add_placeholder()
        self.config_lr = 0.001
        # self.dropout=0.5 # conflict with the placeholder from add_placeholder()
        self.config_dropout = 0.5
        self.nepochs = 20
        self.batch_size = 5
        self.lr_decay = 0.9
        self.nepoch_no_imprv = 3

        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s')
        self.logger = logger

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")  # batch size, max length of sentence in batch
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")  # shape = batch size
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        word_ids, sequence_lengths = pad_sequences(words, 0)
        # feed = {
        #     self.word_ids: word_ids,
        #     self.sequence_lengths: sequence_lengths
        # }
        # if labels is not None:
        #     labels, _ = pad_sequences(labels, 0)
        # feed[self.labels] = labels
        # if lr is not None:
        #     feed[self.lr] = lr
        # if dropout is not None:
        #     feed[self.dropout] = dropout
        lab,length=pad_sequences(labels,0)
        # for l in lab:
        #     print(len(l))

        # feed = {self.word_ids: word_ids, self.labels: lab, self.lr: lr, self.dropout: dropout,
        #         self.sequence_lengths: sequence_lengths}

        feed = {self.word_ids: word_ids, self.labels: lab, self.lr: self.config_lr, self.dropout: self.config_dropout,
                self.sequence_lengths:[89]}

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=False)

        word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
                                                 name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell,
                                                                        lstm_cell, self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.config_dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2 * self.hidden_size, self.ntags],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

    def add_pred_op(self):
        """
        Adds labels_pred to self
        """

        if not self.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """
        Adds loss to self
        """

        if self.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def add_train_op(self):
        """
        Add train_op to self
        """

        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.graph_path, sess.graph)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    def predict_batch(self, sess, words):
        """
        Args:
        sess: a tensorflow session
        words: list of sentences
        Returns:
        labels_pred: list of labels for each sentences.txt
        sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        if self.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, transition_params)
                viterbi_sequences += [viterbi_sequence]
                return viterbi_sequences, sequence_lengths
        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths

    def run_epoch(self, sess, train, dev, tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
        sess: tensorflow session
        train: dataset that yields tuple of sentences, tags
        dev: dataset
        tags: {tag: index} dictionary
        epoch: (int) number of the epoch
        """
        nbatches = (len(train) + self.batch_size - 1) / self.batch_size
        for i, (words, labels) in enumerate(minibatches(train, self.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.lr, self.dropout)
            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)
        acc, f1 = self.run_evaluate(sess, dev, tags)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, f1

    def run_evaluate(self, sess, test, tags):
        """
        Evaluates performance on test set
        Args:
        sess: tensorflow sessionb
        test: dataset that yields tuple of sentences, tags
        tags: {tag: index} dictionary
        Returns:
        accuracy
        f1 score
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += map(lambda element: element[0] == element[1], zip(lab, lab_pred))
                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1

    def train(self, train, dev, tags):
        """
        Performs training with early stopping and lr exponential decay
        Args:
        train: dataset that yields tuple of sentences, tags
        dev: dataset
        tags: {tag: index} dictionary
        """

        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_output)
            # restore session
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt)
                saver.restore(sess, self.model_output)
            else:
                print("Begin to initialize ...")
                sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.nepochs))
                acc, f1 = self.run_epoch(sess, train, dev, tags, epoch)
                # decay learning rate
                self.lr *= self.lr_decay
                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.model_output):
                        os.makedirs(self.model_output)
                    saver.save(sess, self.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                            nepoch_no_imprv))
                    break

    def evaluate(self, test, tags):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing softmax_model over test set")
            saver.restore(sess, self.model_output)
            acc, f1 = self.run_evaluate(sess, test, tags)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))


if __name__ == '__main__':
    # load vocabs
    vocab_words = load_vocab("./preprocess/data/vocab.txt")
    vocab_tags = {"O": 0, "Ori-S": 1, "Ori-I": 2, "Ori-E": 3}

    # print(vocab_words)
    # get processing functions
    processing_word = get_processing_word(vocab_words,
                                          lowercase=True)
    processing_tag = get_processing_word(vocab_tags, lowercase=False)

    # print(processing_word('集'))  # 267
    # print(processing_tag('Ori-S'))  # 1

    # get pre trained embeddings
    w2v_model = Word2Vec.load("./preprocess/softmax_model/w2v")
    vocabList = w2v_model.wv.index2word  # get the list of vocabulary from the preprocess softmax_model
    embeddings = w2v_model[vocabList]  # 2-dimensional array shaped (315, 100)

    # create dataset
    dev = CoNLLDataset("./preprocess/data/data.val", processing_word,
                       processing_tag, None)
    test = CoNLLDataset("./preprocess/data/data.test", processing_word,
                        processing_tag, None)
    train = CoNLLDataset("./preprocess/data/data.train", processing_word,
                         processing_tag, None)

    # get logger
    logger = None

    # build softmax_model
    model = NERModel(embeddings, ntags=len(vocab_tags),
                     logger=logger)
    model.build()
    # train, evaluate and interact
    model.train(train, dev, vocab_tags)

