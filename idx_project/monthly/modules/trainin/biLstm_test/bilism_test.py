import numpy as np
import tensorflow as tf

# 参数设置
graph_path = "./graph"
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# 构建随机特征
x = np.random.rand(num_examples, num_words).astype(np.float32)
x=(x*100).astype(np.int32)

# 构建随机tag
y = np.random.randint(
    num_tags, size=[num_examples, num_words]).astype(np.int32)

train_dataset= tf.data.Dataset.from_tensor_slices((x,y))
train_dataset=train_dataset.batch(2)

iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
train_initializer=iterator.make_initializer(train_dataset)

with tf.variable_scope('inputs'):
    with tf.Session() as sess:
        sess.run(tf)