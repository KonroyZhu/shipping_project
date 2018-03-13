import numpy as np
import tensorflow as tf
import pickle

with open("../../data/processed/data_for_words.pkl", 'rb') as f:
    x, y, word2idx, tag2index, idx2word = pickle.load(f)

# Data settings.
num_examples = x.shape[0]
batch_size = 10
num_words = x.shape[1]
num_features = x.shape[2]
num_tags = len(tag2index)

# Random features.
# x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)
#
# # Random tag indices representing the gold sequence.
# y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# All sequences in this example have the same length, but they can be variable in a real softmax_model.
sequence_lengths = np.full(batch_size, num_words - 1, dtype=np.int32)


def get_next(x, y, step):
    locate = batch_size * step
    next_x = x[locate:locate + batch_size]
    next_y = y[locate:locate + batch_size]
    return next_x, next_y

sub=3
stud=4
l=[[60,90,80],[90,91,91],[80,70,70],[90,30,50]]
for i in range(sub):
    sum=0
    for j in range(stud):
        sum+=l[j][i]
    print(sum/stud)


'''
# Train and evaluate the softmax_model.
with tf.Graph().as_default():
    with tf.Session() as session:
        # Add the data to the TensorFlow graph.
        # x_t=tf.constant(x)
        # y_t = tf.constant(y)
        x_t = tf.placeholder(dtype=tf.float32,shape=[batch_size,num_words,num_features])
        y_t=tf.placeholder(dtype=tf.int32,shape=[batch_size,num_words])
        sequence_lengths_t = tf.constant(sequence_lengths)

        # Compute unary scores from a linear layer.
        weights = tf.get_variable("weights", [num_features, num_tags])
        matricized_x_t = tf.reshape(x_t, [-1, num_features])
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)
        print(num_examples,"  "+str(num_words)," "+str(num_tags))
        unary_scores = tf.reshape(matricized_unary_scores,
                                  [batch_size, num_words, num_tags])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            unary_scores, y_t, sequence_lengths_t)

        # Add a training op to tune the parameters.
        loss = tf.reduce_mean(-log_likelihood)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # Train for a fixed number of iterations.
        session.run(tf.global_variables_initializer())
        # for step in range(int(num_examples / batch_size)):
        for step in range(1):
            x1, y1 = get_next(x, y, step)
            for i in range(1000):
                tf_unary_scores, tf_transition_params, _ = session.run(
                    [unary_scores, transition_params, train_op], feed_dict={x_t: x1, y_t: y1})
                if i % 100 == 0:
                    correct_labels = 0
                    total_labels = 0
                    for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y,
                                                                      sequence_lengths):
                        # Remove padding from the scores and tag sequence.
                        tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                        y_ = y_[:sequence_length_]

                        # Compute the highest scoring sequence.
                        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                            tf_unary_scores_, tf_transition_params)

                        # Evaluate word-level accuracy.
                        correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                        total_labels += sequence_length_
                        print(viterbi_sequence)
                        print(y_)
                    accuracy = 100.0 * correct_labels / float(total_labels)
                    print("Accuracy: %.2f%%" % accuracy)



                    # '''
