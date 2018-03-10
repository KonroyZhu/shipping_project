import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
from os.path import join


class Bilstm_Model:
    def __init__(self):
        # config part 1
        self.batch_size = 50
        self.train_batch_size = 50
        self.dev_batch_size = 50

        self.embedding_size = 100
        self.keep_prob = 0.5
        self.num_unit = 100
        self.num_layer = 2
        self.learning_rate = 0.01
        self.time_step = 194  # TODO: data_util need modification on  a fixed time step

        # load data
        self.sen_list, self.tag_list, self.char2index, self.tag2index, self.index2char, self.index2tag = self.load_data()
        self.train_x, self.train_y, self.dev_x, self.dev_y, self.test_x, self.test_y = self.split_data(self.sen_list,
                                                                                                       self.tag_list)
        # data set iterator
        self.iterator, self.train_initializer, self.dev_initializer, self.test_initializer = self.iterator_initializer()

        # config part 2
        self.vocab_size = len(self.char2index)
        self.category_num = len(self.tag2index)

    def m_print(self, x, option=1, common=""):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(self.train_initializer)
            common = "  " + common
            if option == 1:
                print(common + str(session.run(x)))
            elif option == 2:
                print(common + str(session.run(tf.shape(x))))

    # load data
    def load_data(self):
        f = open("../../data/processed/data.pkl", 'rb')
        sen_list, tag_list, char2index, tag2index, index2char, index2tag = pickle.load(f)
        return sen_list, tag_list, char2index, tag2index, index2char, index2tag

    def split_data(self, x, y):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=40)
        train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40)
        return train_x, train_y, dev_x, dev_y, test_x, test_y

    def iterator_initializer(self):
        # train and dev dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        train_dataset = train_dataset.batch(batch_size=self.batch_size)

        dev_dataset = tf.data.Dataset.from_tensor_slices((self.dev_x, self.dev_y))
        dev_dataset = dev_dataset.batch(batch_size=self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))
        test_dataset = test_dataset.batch(batch_size=self.batch_size)

        # A reinitializeable iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        train_initializer = iterator.make_initializer(train_dataset)
        dev_initializer = iterator.make_initializer(dev_dataset)
        test_initializer = iterator.make_initializer(test_dataset)

        return iterator, train_initializer, dev_initializer, test_initializer

    # embedding
    def embedding(self):
        # Input layer
        with tf.variable_scope('input'):
            x, y_label = self.iterator.get_next()
            # tf_print(x,2)

        # Embedding layer
        with tf.variable_scope('embedding'):
            embedding = tf.Variable(tf.random_normal([self.vocab_size + 1, self.embedding_size]), dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x)
        return inputs, y_label

    # weight
    def weight(self, shape, stddev=0.1, mean=0):
        initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)

    # and bias
    def bias(self, shape, value=0.1):
        initial = tf.constant(value=value, shape=shape)
        return tf.Variable(initial)

    # bi-lstm softmax_model
    def bi_lstm(self, inputs):
        with tf.variable_scope('bi-lstm'):
            # Bi-Lstm layer
            print("# bi_lstm")
            inputs = tf.unstack(inputs, self.time_step, axis=1)  # reshape input to suit the api
            cell_fw = [tf.contrib.rnn.LSTMCell(self.num_unit, self.keep_prob) for _ in range(self.num_layer)]
            cell_bw = [tf.contrib.rnn.LSTMCell(self.num_unit, self.keep_prob) for _ in range(self.num_layer)]

            output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
            self.m_print(output, 2, "output shape: ")
            output = tf.stack(output, axis=1)
            self.m_print(output, 2, "output shape after tf.stack: ")
            output = tf.reshape(output, [-1, self.num_unit * 2])
            self.m_print(output, 2, "output shape after reshape: ")

            return output

    def predict(self, output):
        with tf.variable_scope('predict'):
            print("# predict")
            w = self.weight([self.num_unit * 2, self.category_num])
            b = self.bias([self.category_num])
            y = tf.matmul(output, w) + b
            y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)

            # printing out and summarizing
            tf.summary.histogram('y_predict', y_predict)  # for tensorboard summary
            self.m_print(y, 2, "y(predict) shape: ")
            self.m_print(y_predict, 2, "reshaped y shape: ")

            return y_predict, y

    def loss(self, y_label, y_predict):
        print("# loss")

        labels = y_label
        predict = tf.cast(y_predict, tf.float32)
        argmax_predict = tf.cast(tf.argmax(predict, axis=1), tf.int32)

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predict))

        # printing out and summarizing
        self.m_print(predict, 1, "y_predict:")
        self.m_print(argmax_predict, 1, "y_predict after argmax:")
        self.m_print(labels, 1, "y_labels:")
        self.m_print(cross_entropy, 1, "cross_entropy:")
        tf.summary.scalar('loss', cross_entropy)

        return cross_entropy

    def train(self):
        print("# training")
        inputs, y_label = self.embedding()  # input: data set, y_label: original labels
        y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)  # y_label_reshape: reshaped y_label
        self.m_print(y_label_reshape, 2, "original label shape: ")
        self.m_print(inputs, 2, "inputs shape: ")
        self.m_print(y_label, 2, "label shape: ")

        output = self.bi_lstm(inputs)  # output: trained lstm output

        y_predict, y = self.predict(output)  # y_predict: reshaped y, y: original y

        cross_entropy = self.loss(y_label_reshape, y)  # cross_entropy: loss

        # accuracy
        correct_prediction = tf.equal(y_predict, y_label_reshape)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Steps
        global_step = tf.Variable(-1, trainable=False, name='global_step')

        # Train
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)

        # training
        with tf.Session() as sess:
            for epoch in range(1000):
                # Train
                sess.run(self.train_initializer)
                loss, acc, _ = sess.run([cross_entropy, accuracy, train]
                                        )
                # Print log
                print("Epoch " + str(epoch) + " in 1000", 'Train Loss', loss, 'Accuracy', acc)


if __name__ == '__main__':
    bilstm_model = Bilstm_Model()
    bilstm_model.train()
