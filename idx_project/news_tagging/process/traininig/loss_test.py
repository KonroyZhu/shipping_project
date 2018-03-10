import tensorflow as tf

# p=  [1,1,1,1,1,1,1,1,1,1,1]#3
# p=  [0,0,0,0,0,0,0,0,0,0,0]#6

p=[1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#31 #10.397207
p=[1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#25 #10.370044
# p=[0.0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#18 #9.332339    #12.903112
# p=[0.0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#25 #10.370044 #15.037716
l=[1,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

labels = tf.constant(l)
predict = tf.constant(p)

loss_more = 5
loss_less = 1

# loss = tf.where(tf.greater(labels, predict), (labels - predict) * loss_more, (predict - labels) * loss_less)
# loss_mean = tf.reduce_mean(loss*10)
loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=predict)
loss_mean=tf.reduce_mean(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(predict))
    print(sess.run(labels))
    print(sess.run(loss))
    print(sess.run(loss_mean))
