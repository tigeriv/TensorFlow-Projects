import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Parameters to adjust size (num_res is either 2 or 3)
layer_sizes = [16, 32, 64, 128]
num_res = 3


def bn_relu(X):
    X = tf.layers.batch_normalization(X)
    X = tf.nn.leaky_relu(X)
    return X


def convolve(X, input_channels, num_filters, stride, name, size=3):
    filters = tf.get_variable(name + 'filters', shape=(size, size, input_channels, num_filters),
                               initializer=tf.keras.initializers.glorot_normal())
    bias = tf.get_variable(name + 'bias', shape=(num_filters,), initializer=tf.zeros_initializer())
    X = tf.nn.conv2d(X, filters, strides=stride, padding="SAME")
    X = tf.nn.bias_add(X, bias)
    return X


def res_unit(num_filters, input_channels, x_in, first_layer=False, stride=2, name=""):
    # Normalize
    if not first_layer:
        X = bn_relu(x_in)
    else:
        X = x_in
    # 3x3 conv, stride 2 to reduce size by half or 1 to keep same
    X = convolve(X, input_channels, num_filters, stride, name + "First")
    # Second convolution, stride = 1
    X = bn_relu(X)
    X = convolve(X, num_filters, num_filters, 1, name + "Sec")
    # Short cut input by 1x1 convoution
    shortcut = convolve(x_in, input_channels, num_filters, stride, name + "Short", size=1)
    shortcut = tf.layers.batch_normalization(shortcut)
    # Combine and return
    return tf.math.add(shortcut, X)


# Multiply H, W by 2 and then merge
def upsample(X, X_skip):
    X = tf.keras.layers.UpSampling2D((2, 2))(X)
    X = tf.concat([X, X_skip], -1)
    return X


class ResUNet:
    def __init__(self, learning_rate=0.007, depth=False, width=1920, height=1080, classes=8):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.depth = 3
        self.width = width
        self.height = height
        self.num_classes = classes
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

            # Residual unit encoding
            # 64 depth
            X1 = res_unit(layer_sizes[0], self.depth, self.X, first_layer=True, stride=1, name="Enc1")
            # 1/2 size
            X2 = res_unit(layer_sizes[1], layer_sizes[0], X1, name="Enc2")
            X3 = X2
            # 1/4 size
            if num_res == 3:
                X3 = res_unit(layer_sizes[2], layer_sizes[1], X2, name="Enc3")

            # Bridge
            # 1/8 size
            X_bridge = res_unit(layer_sizes[num_res], layer_sizes[num_res-1], X3, name="Bridge")

            # Decode
            if num_res == 3:
                X_D3 = upsample(X_bridge, X3)
                X_D3 = res_unit(layer_sizes[2], (layer_sizes[2] + layer_sizes[3]), X_D3, stride=1, name="Dec1")
            else:
                X_D3 = X_bridge

            X_D2 = upsample(X_D3, X2)
            # 1/2 size
            X_D2 = res_unit(layer_sizes[1], (layer_sizes[1] + layer_sizes[2]), X_D2, stride=1, name="Dec2")
            # Full size
            X_D1 = upsample(X_D2, X1)
            X_D1 = res_unit(layer_sizes[0], (layer_sizes[0] + layer_sizes[1]), X_D1, stride=1, name="Dec3")

            # Make pixel-wise predictions
            predictions = convolve(X_D1, layer_sizes[0], self.num_classes, 1, "Predictions", size=1)
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=self.labels)
            loss = tf.reduce_mean(cost)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
        return predictions, cost, loss, train_op, init, optimizer, saver

    def get_size(self):
        if self.depth:
            return self.width, self.height, 4
        else:
            return self.width, self.height, 3


if __name__ == "__main__":
    model = ResUNet()
    print(model.predictions.shape)
