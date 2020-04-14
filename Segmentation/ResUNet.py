import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def bn_relu(X):
    X = tf.layers.batch_normalization(X)
    X = tf.nn.leaky_relu(X)
    return X


def convolve(X, input_channels, num_filters, stride, size=3):
    filters = tf.get_variable('filters', shape=(size, size, input_channels, num_filters),
                               initializer=tf.keras.initializers.glorot_normal())
    bias = tf.get_variable('bias', shape=(num_filters,), initializer=tf.zeros_initializer())
    X = tf.nn.conv2d(X, filters, strides=stride, padding="VALID")
    X = tf.nn.bias_add(X, bias)
    return X


def res_unit(num_filters, input_channels, x_in, first_layer=False, stride=2):
    # Normalize
    if not first_layer:
        X = bn_relu(x_in)
    else:
        X = x_in
    # 3x3 conv, stride 2 to reduce size by half or 1 to keep same
    X = convolve(X, input_channels, num_filters, stride)
    # Second convolution, stride = 1
    X = bn_relu(X)
    X = convolve(X, num_filters, num_filters, 1)
    # Short cut input by 1x1 convoution
    shortcut = convolve(x_in, input_channels, num_filters, stride, size=1)
    shortcut = tf.layers.batch_normalization(shortcut)
    # Combine and return
    return tf.math.add(shortcut, X)


# Multiply H, W by 2 and then merge
def upsample(X, X_skip):
    X = tf.keras.layers.UpSampling2D((2, 2))(X)
    cX= tf.concat([X, X_skip], -1)
    return X


class ResUNet:
    def __init__(self, learning_rate=0.007, depth=False, width=1920, height=1080, classes=20):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.depth = 3
        self.width = width
        self.height = height
        if depth:
            self.depth += 1
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, (None, self.height, self.width, self.depth))
            self.labels = tf.placeholder(tf.float32, (None, self.height, self.width, 1))
        self.training = True
        self.predictions, self.cost, self.loss, self.train_op, self.init, self.optimizer = self.make_graph(learning_rate)

    def make_graph(self, learning_rate):
        with self.graph.as_default():
            # filter shape: filter_height, filter_width, in_channels, out_channels

            # Residual unit encoding
            # 64 depth
            with tf.name_scope("enc1"):
                X1 = res_unit(64, self.depth, self.X, first_layer=True, stride=1)
            # 1/2 size, 128 depth
            with tf.name_scope("enc2"):
                X2 = res_unit(128, 64, X1)
            # 1/4 size, 256 depth
            with tf.name_scope("enc3"):
                X3 = res_unit(256, 128, X2)

            # Bridge
            # 1/8 size, 512 depth
            with tf.name_scope("bridge"):
                X_bridge = res_unit(512, 256, X3)

            # Decode
            with tf.name_scope("dec1"):
                # (512 + 256) input depth
                X_D3 = upsample(X_bridge, X3)
                # 1/4 size, 256 depth
                X_D3 = res_unit(256, (512 + 256), X_D3, stride=1)
            with tf.name_scope("dec2"):
                # (256 + 128) input depth
                X_D2 = upsample(X_D3, X2)
                # 1/2 size, 128 depth
                X_D2 = res_unit(128, (256 + 128), X_D2, stride=1)
            with tf.name_scope("dec3"):
                # (128 + 64) input depth
                X_D1 = upsample(X_D2, X1)
                # 64 depth
                X_D1 = res_unit(64, (128 + 64), X_D1, stride=1)


            with tf.name_scope("fcn1"):
                W1 = tf.get_variable('W1', shape=(9216, 4096), initializer=tf.keras.initializers.glorot_normal())
                b1 = tf.Variable(tf.zeros((4096,)), trainable=True)
                X6 = tf.matmul(X5, W1) + b1
                X6 = tf.layers.batch_normalization(X6)
                X6 = tf.nn.leaky_relu(X6)
            with tf.name_scope("fcn2"):
                W2 = tf.get_variable('W2', shape=(4096, 4096), initializer=tf.keras.initializers.glorot_normal())
                b2 = tf.Variable(tf.zeros((4096,)), trainable=True)
                X7 = tf.matmul(X6, W2) + b2
                X7 = tf.layers.batch_normalization(X7)
                X7 = tf.nn.leaky_relu(X7)
            with tf.name_scope("softmax"):
                W3 = tf.get_variable('W3', shape=(4096, 10), initializer=tf.keras.initializers.glorot_normal())
                b3 = tf.Variable(tf.zeros((10,)), trainable=True)
                X8 = tf.add(tf.matmul(X7, W3), b3)
                if self.training:
                    X8 = tf.nn.dropout(X8, rate=0.2)

            predictions = tf.nn.softmax(X8)

            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=self.labels)
            loss = tf.reduce_mean(cost)

            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            train_op = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
        return predictions, cost, loss, train_op, init, optimizer

    def get_size(self):
        return 227, 227, 3
