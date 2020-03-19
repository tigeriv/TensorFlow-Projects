from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import pickle
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LeNet:
    def __init__(self, learning_rate=0.007):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, (None, 32, 32, 3))
            self.labels = tf.placeholder(tf.int32, (None,))
        self.training = True
        self.predictions, self.cost, self.loss, self.train_op, self.init, self.optimizer = self.make_graph(learning_rate)

    def make_graph(self, learning_rate):
        with self.graph.as_default():
            # filter shape: filter_height, filter_width, in_channels, out_channels
            with tf.name_scope("conv1"):
                filters1 = tf.get_variable('filters1', shape=(5, 5, 3, 6),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias1 = tf.get_variable('bias1', shape=(6,), initializer=tf.zeros_initializer())
                X1 = tf.nn.conv2d(self.X, filters1, strides=1, padding="VALID")
                X1 = tf.nn.bias_add(X1, bias1)
                X1 = tf.layers.batch_normalization(X1)
                X1 = tf.nn.leaky_relu(X1)
                X1 = tf.nn.avg_pool2d(X1, ksize=2, strides=2, padding="VALID")
            with tf.name_scope("conv2"):
                filters2 = tf.get_variable('filters2', shape=(5, 5, 6, 16),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias2 = tf.get_variable('bias2', shape=(16,), initializer=tf.zeros_initializer())
                X2 = tf.nn.conv2d(X1, filters2, strides=1, padding="VALID")
                X2 = tf.nn.bias_add(X2, bias2)
                X2 = tf.layers.batch_normalization(X2)
                X2 = tf.nn.leaky_relu(X2)
                X2 = tf.nn.avg_pool2d(X2, ksize=2, strides=2, padding="VALID")
                X2 = tf.layers.flatten(X2)
            with tf.name_scope("fcn1"):
                W1 = tf.get_variable('W1', shape=(400, 120), initializer=tf.keras.initializers.glorot_normal())
                b1 = tf.Variable(tf.zeros((120,)), trainable=True)
                X3 = tf.matmul(X2, W1) + b1
                X3 = tf.layers.batch_normalization(X3)
                X3 = tf.nn.leaky_relu(X3)
            with tf.name_scope("fcn2"):
                W2 = tf.get_variable('W2', shape=(120, 84), initializer=tf.keras.initializers.glorot_normal())
                b2 = tf.Variable(tf.zeros((84,)), trainable=True)
                X4 = tf.matmul(X3, W2) + b2
                X4 = tf.layers.batch_normalization(X4)
                X4 = tf.nn.leaky_relu(X4)
            with tf.name_scope("softmax"):
                W3 = tf.get_variable('W3', shape=(84, 10), initializer=tf.keras.initializers.glorot_normal())
                b3 = tf.Variable(tf.zeros((10,)), trainable=True)
                X5 = tf.add(tf.matmul(X4, W3), b3)
                if self.training:
                    X5 = tf.nn.dropout(X5, rate=0.2)

            predictions = tf.nn.softmax(X5)

            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=self.labels)
            loss = tf.reduce_mean(cost)

            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            train_op = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
        return predictions, cost, loss, train_op, init, optimizer

    def get_size(self):
        return 32, 32, 3