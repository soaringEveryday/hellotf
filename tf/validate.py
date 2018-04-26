import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('hello world')
session = tf.Session()
print(session.run(hello))