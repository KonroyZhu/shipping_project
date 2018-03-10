import tensorflow as tf

from tensorflow.python.framework import ops
word_ids=tf.placeholder(tf.float32,shape=[None,None])
fd={word_ids:[[1,2,3,4]]}
print(fd)
print(isinstance(fd, ops.Tensor))
with tf.Session() as sess:
    print(sess.run(word_ids,{word_ids:[[1,2,3,4]]}))
