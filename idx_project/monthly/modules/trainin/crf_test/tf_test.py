import tensorflow as tf
import numpy as np

x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # e (3, 3)
y = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)

x_t = tf.constant(x, dtype=tf.float32)
y_t = tf.constant(y, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mul = sess.run(tf.matmul(x_t, y_t))  # shape=[3,2] 3*2=6
    re_mul = tf.reshape(mul, [1, 6])  # shape=[1,6] 1*6=6
    print(mul)
    print(sess.run(re_mul))
