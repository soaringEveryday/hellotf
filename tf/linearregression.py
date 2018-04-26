import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# actual input x
x = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

# formula
y = tf.matmul(x, W) + b

# actual output y
y_ = tf.placeholder(tf.float32, [None, 1])
# SSE
cost = tf.reduce_sum(tf.pow((y_-y), 2))

# fake data and start train
learn_rate = 0.00001
learn_count = 200
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
sess = tf.Session()
# must do
sess.run(tf.global_variables_initializer())

for i in range(learn_count):
    xs = np.array([[i]])
    ys = np.array([[2*i]])

    feed = {x: xs, y_: ys}
    sess.run(train_step, feed_dict=feed)

    print("after %d iteration: " % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))

# # batch train
# for i in range(learn_count):
#     x_train = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
#     y_train = [[10], [11.5], [12], [13], [14.5], [15.5], [16.8], [17.3], [18], [18.7]]
#
#     feed = {x: x_train, y_: y_train}
#     sess.run(train_step, feed_dict=feed)
#
#     print("after %d iteration: " % i)
#     print("W: %f" % sess.run(W))
#     print("b: %f" % sess.run(b))