import numpy as np
import tensorflow as tf

# 参数设置
graph_path = "./graph"
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# 构建随机特征
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

# 构建随机tag
y = np.random.randint(
    num_tags, size=[num_examples, num_words]).astype(np.int32)



# 获取样本句长向量（因为每一个样本可能包含不一样多的词），在这里统一设为 num_words - 1，真实情况下根据需要设置
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

# print(x)
print(x.shape)
print(y.shape)
print(y)
"""
# 训练，评估模型
with tf.Graph().as_default():
    with tf.Session() as session:
        x_t = tf.constant(x)  # [10 20 100]
        y_t = tf.constant(y)  # [10 20]
        sequence_lengths_t = tf.constant(sequence_lengths)  # [19 19 19 19 19 19 19 19 19 19]

        # 在这里设置一个无偏置的线性层
        weights = tf.get_variable("weights", [num_features, num_tags])  # [100 5] randomly created matrix for weight
        matricized_x_t = tf.reshape(x_t, [-1, num_features])  # [10 20 100] -> [10*20 100] = [200 100]
        matricized_unary_scores = tf.matmul(matricized_x_t,
                                            weights)  # [200   5]   result of matricized_x_t * weight
        unary_scores = tf.reshape(matricized_unary_scores,
                                  [num_examples, num_words, num_tags])  # [10 20  5]

        # 计算log-likelihood并获得transition_params
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            unary_scores, y_t, sequence_lengths_t)  # log_likelihood: [10], transition_params: [5,5]

        # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            unary_scores, transition_params, sequence_lengths_t)  # viterbi_sequence: prediction, viterbi_score:

        loss = tf.reduce_mean(-log_likelihood)  # average value of log_likelihood
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        session.run(tf.global_variables_initializer())

        session.run(tf.global_variables_initializer())

        mask = (np.expand_dims(np.arange(num_words), axis=0) <  # np.arange()创建等差数组
                np.expand_dims(sequence_lengths, axis=1))  # np.expand_dims()扩张维度

        # 得到一个num_examples*num_words的二维数组，数据类型为布尔型，目的是对句长进行截断

        # 将每个样本的sequence_lengths加起来，得到标签的总数
        total_labels = np.sum(sequence_lengths)

        # 进行训练
        for i in range(1000):
            tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op])
            if i % 100 == 0:
                correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)

                # sequence for line 0
                print(y[0])
                print(session.run(viterbi_sequence[0]))

        #   testing
        length = tf.constant([19])
        x_test = x_t[0] # sample vector for a sentences.txt
        x_test_weigh = tf.matmul(x_test, weights)  # multiply the x_test vector to the trained weight vector
        x_test_weigh_re = tf.reshape(x_test_weigh, [1, num_words, num_tags])  # reshape it to feed the decoder
        sequence_test, _ = tf.contrib.crf.crf_decode(
            x_test_weigh_re, transition_params, length)  # predict tags sequence by the decoder
        print(session.run(sequence_test))
        """
