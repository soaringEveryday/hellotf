import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input x dimension
n = 2

# actual input x
x = tf.placeholder(tf.float32, [None, n])

W = tf.Variable(tf.zeros([n, 1]))
b = tf.Variable(tf.zeros([1]))

# formula
y = tf.matmul(x, W) + b

# actual output y
y_ = tf.placeholder(tf.float32, [None, 1])

# make cost (SSE)
cost = tf.reduce_sum(tf.pow((y_-y), 2))

# fake data and start train
learn_rate = 0.00001
learn_count = 200
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
sess = tf.Session()
# must do
sess.run(tf.global_variables_initializer())

for i in range(learn_count):
    x_train = [[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]]
    y_train = [[7], [8], [10], [14], [8], [13], [20], [16], [28], [26]]

    feed = {x: x_train, y_: y_train}
    sess.run(train_step, feed_dict=feed)

    print("after %d iteration: " % i)
    print("W0: %f" % sess.run(W[0]))
    print("W1: %f" % sess.run(W[1]))
    print("b: %f" % sess.run(b))
