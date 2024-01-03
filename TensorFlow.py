# TensorFlow 이해

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)

sess = tf.Session()
print(sess.run(node3))