import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()


def activation(X):
    X = tf.layers.batch_normalization(X)
    X = tf.nn.leaky_relu(X)
    return X


def conv2d(X, input_channels, num_filters, stride, name, size=3):
    X = tf.keras.layers.Conv2D(num_filters, size, stride, padding='same',
                               kernel_initializer=tf.keras.initializers.glorot_normal(), bias_initializer=tf.zeros_initializer(),
                               use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.00004))(X)
    # filters = tf.get_variable(name + 'filters', shape=(size, size, input_channels, num_filters),
    #                            initializer=tf.keras.initializers.glorot_normal())
    # bias = tf.get_variable(name + 'bias', shape=(num_filters,), initializer=tf.zeros_initializer())
    # X = tf.nn.conv2d(X, filters, strides=stride, padding="SAME")
    # X = tf.nn.bias_add(X, bias)
    return X


# Depthwise Conv 2D
def DWConv(X, input_channels, multiplier, stride, name, size=3, dilations=None):
    filters = tf.get_variable(name + 'filters', shape=(size, size, input_channels, multiplier),
                               initializer=tf.keras.initializers.glorot_normal())
    bias = tf.get_variable(name + 'bias', shape=(input_channels * multiplier,), initializer=tf.zeros_initializer())
    X = tf.nn.depthwise_conv2d(X, filters, strides=(1, stride, stride, 1), padding="SAME", dilations=dilations)
    X = tf.nn.bias_add(X, bias)
    return X


# Depthwise Separable Convolution
def DSConv(X, num_outs, stride, size=3):
    X = tf.layers.separable_conv2d(X, num_outs, size, strides=stride, padding='same')
    return X


def bottleneck(X, in_channels, expansion, num_outs, stride, name, repeated=3):
    X = conv2d(X, in_channels, expansion*in_channels, 1, name + "Conv1", size=1)
    X = activation(X)
    X = DWConv(X, expansion*in_channels, 1, stride, name + "DWConv")
    X = activation(X)
    X = conv2d(X, expansion*in_channels, num_outs, 1, name + "Conv2", size=1)
    return X


# Multiply H, W by 2 and then merge
def FFM(X_deep, X_shallow, multiplier):
    X_deep = tf.keras.layers.UpSampling2D((multiplier, multiplier))(X_deep)
    X_deep = activation(X_deep)
    X_deep = DWConv(X_deep, 128, 1, 1, "XDeep", dilations=(multiplier, multiplier))

    X_deep = conv2d(X_deep, 128, 128, 1, "FFMDeep", size=1)
    X_shallow = conv2d(X_shallow, 64, 128, 1, "FFMShallow", size=1)

    X = tf.keras.layers.add([X_deep, X_shallow])
    X = activation(X)
    return X


# Pyramid pooling module
def PPM(X, bin_sizes, width, height, num_outs):
    concat_list = []

    for bin_size in bin_sizes:
        bin_X = tf.keras.layers.AveragePooling2D(pool_size=(width // bin_size, height // bin_size), strides=(width // bin_size, height // bin_size))(X)
        bin_X = tf.keras.layers.Conv2D(num_outs//(len(bin_sizes)), 3, strides=2, padding='same')(bin_X)
        bin_X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (height, width)))(bin_X)
        concat_list.append(bin_X)

    return tf.keras.layers.concatenate(concat_list)


class FSCNN:
    def __init__(self, learning_rate=0.007, depth=False, width=1920, height=1080, classes=8, training=True):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.depth = 3
        self.width = width
        self.height = height
        self.num_classes = classes
        self.training = training
        if depth:
            self.depth += 1
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, (None, None, None, self.depth))
            self.labels = tf.placeholder(tf.int32, (None, None, None))
        self.training = True
        self.predictions, self.cost, self.loss, self.train_op, self.init, self.optimizer, self.saver = self.make_graph(learning_rate)

    def make_graph(self, learning_rate):
        with self.graph.as_default():
            # filter shape: filter_height, filter_width, in_channels, out_channels

            # Both Branches
            # Conv 1
            X = conv2d(self.X, self.depth, 32, 2, "Conv1")
            X = activation(X)
            # DSConv1
            X = DSConv(X, 48, 2)
            X = activation(X)
            # DSConv2
            X = DSConv(X, 64, 2)
            X = activation(X)

            # Branch off, activation is called within bottleneck
            # Bottleneck 1
            X_deep = bottleneck(X, 64, 6, 64, 2, "BN1")
            # Bottleneck 2
            X_deep = bottleneck(X_deep, 64, 6, 96, 2, "BN2")
            # Bottleneck 3
            X_deep = bottleneck(X_deep, 96, 6, 128, 1, "BN3")
            # PPM (skip this to allow variable sized input)
            # X_deep = PPM(X_deep, [2, 4, 6, 8], self.width//32, self.height//32, 128)
            # FFM
            X = FFM(X_deep, X, 4)

            # Classify
            X = DSConv(X, 128, 1)
            X = DSConv(X, 128, 1)
            predictions = conv2d(X, 128, self.num_classes, 1, "Predictions", size=1)
            predictions = tf.keras.layers.UpSampling2D((8, 8))(predictions)
            if self.training:
                predictions = tf.nn.dropout(predictions, rate=0.2)

            # Compute Cost
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=self.labels)
            loss = tf.reduce_mean(cost)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
        return predictions, cost, loss, train_op, init, optimizer, saver


if __name__ == "__main__":
    model = FSCNN(width=2048, height=1024)
    print(model.predictions.shape)
