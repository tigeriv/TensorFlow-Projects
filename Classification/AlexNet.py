import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class AlexNet:
    def __init__(self, learning_rate=0.007):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, (None, 227, 227, 3))
            self.labels = tf.placeholder(tf.int32, (None,))
        self.training = True
        self.predictions, self.cost, self.loss, self.train_op, self.init, self.optimizer = self.make_graph(learning_rate)

    def make_graph(self, learning_rate):
        with self.graph.as_default():
            # filter shape: filter_height, filter_width, in_channels, out_channels
            with tf.name_scope("conv1"):
                filters1 = tf.get_variable('filters1', shape=(11, 11, 3, 96),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias1 = tf.get_variable('bias1', shape=(96,), initializer=tf.zeros_initializer())
                X1 = tf.nn.conv2d(self.X, filters1, strides=4, padding="VALID")
                X1 = tf.nn.bias_add(X1, bias1)
                X1 = tf.layers.batch_normalization(X1)
                X1 = tf.nn.leaky_relu(X1)
                X1 = tf.nn.max_pool(X1, ksize=3, strides=2, padding="VALID")
            with tf.name_scope("conv2"):
                filters2 = tf.get_variable('filters2', shape=(5, 5, 96, 256),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias2 = tf.get_variable('bias2', shape=(256,), initializer=tf.zeros_initializer())
                X2 = tf.nn.conv2d(X1, filters2, strides=1, padding="SAME")
                X2 = tf.nn.bias_add(X2, bias2)
                X2 = tf.layers.batch_normalization(X2)
                X2 = tf.nn.leaky_relu(X2)
                X2 = tf.nn.max_pool(X2, ksize=3, strides=2, padding="VALID")
            with tf.name_scope("conv3"):
                filters3 = tf.get_variable('filters3', shape=(3, 3, 256, 384),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias3 = tf.get_variable('bias3', shape=(384,), initializer=tf.zeros_initializer())
                X3 = tf.nn.conv2d(X2, filters3, strides=1, padding="SAME")
                X3 = tf.nn.bias_add(X3, bias3)
                X3 = tf.layers.batch_normalization(X3)
                X3 = tf.nn.leaky_relu(X3)
            with tf.name_scope("conv4"):
                filters4 = tf.get_variable('filters4', shape=(3, 3, 384, 384),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias4 = tf.get_variable('bias4', shape=(384,), initializer=tf.zeros_initializer())
                X4 = tf.nn.conv2d(X3, filters4, strides=1, padding="SAME")
                X4 = tf.nn.bias_add(X4, bias4)
                X4 = tf.layers.batch_normalization(X4)
                X4 = tf.nn.leaky_relu(X4)
            with tf.name_scope("conv5"):
                filters5 = tf.get_variable('filters5', shape=(3, 3, 384, 256),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias5 = tf.get_variable('bias5', shape=(256,), initializer=tf.zeros_initializer())
                X5 = tf.nn.conv2d(X4, filters5, strides=1, padding="SAME")
                X5 = tf.nn.bias_add(X5, bias5)
                X5 = tf.layers.batch_normalization(X5)
                X5 = tf.nn.leaky_relu(X5)
                X5 = tf.nn.max_pool(X5, ksize=3, strides=2, padding="VALID")
                X5 = tf.layers.flatten(X5)
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
