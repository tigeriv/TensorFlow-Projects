import cv2
import scipy.io
import numpy as np
from WaymoData import WaymoData
import selective_search


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.enable_eager_execution()


class RCNN:
    def __init__(self, learning_rate=0.007, shape=(1920, 1920), vgg_path="pretrained_model/imagenet-vgg-verydeep-19.mat"):
        self.path = vgg_path
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, (None, shape[0], shape[1], 3))
            self.labels = tf.placeholder(tf.int32, (None, None, 4))
        self.training = True
        self.init, self.features, self.indices = self.make_graph(learning_rate)
        self.batch_size = -1

    # Load Feature Encoder
    # Output features are (batch, 60, 60, 512) for a 1920 x 1920 image
    def load_vgg(self):
        vgg = scipy.io.loadmat(self.path)

        vgg_layers = vgg['layers']

        # Returns weights, bias from layer
        def _weights(layer, expected_layer_name):
            wb = vgg_layers[0][layer][0][0][2]
            W = wb[0][0]
            b = wb[0][1]
            layer_name = vgg_layers[0][layer][0][0][0][0]
            assert layer_name == expected_layer_name
            return W, b

        # Performs a conv2d with ReLu activation
        def _conv2d_relu(prev_layer, layer, layer_name):
            W, b = _weights(layer, layer_name)
            W = tf.constant(W)
            b = tf.constant(np.reshape(b, (b.size)))
            return tf.nn.relu(tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

        # Performs average pool operation
        def _avgpool(prev_layer):
            return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv1_1 = _conv2d_relu(self.X, 0, 'conv1_1')
        conv1_2 = _conv2d_relu(conv1_1, 2, 'conv1_2')
        avgpool1 = _avgpool(conv1_2)

        conv2_1 = _conv2d_relu(avgpool1, 5, 'conv2_1')
        conv2_2 = _conv2d_relu(conv2_1, 7, 'conv2_2')
        avgpool2 = _avgpool(conv2_2)

        conv3_1 = _conv2d_relu(avgpool2, 10, 'conv3_1')
        conv3_2 = _conv2d_relu(conv3_1, 12, 'conv3_2')
        conv3_3 = _conv2d_relu(conv3_2, 14, 'conv3_3')
        conv3_4 = _conv2d_relu(conv3_3, 16, 'conv3_4')
        avgpool3 = _avgpool(conv3_4)

        conv4_1 = _conv2d_relu(avgpool3, 19, 'conv4_1')
        conv4_2 = _conv2d_relu(conv4_1, 21, 'conv4_2')
        conv4_3 = _conv2d_relu(conv4_2, 23, 'conv4_3')
        conv4_4 = _conv2d_relu(conv4_3, 25, 'conv4_4')
        avgpool4 = _avgpool(conv4_4)

        conv5_1 = _conv2d_relu(avgpool4, 28, 'conv5_1')
        conv5_2 = _conv2d_relu(conv5_1, 30, 'conv5_2')
        conv5_3 = _conv2d_relu(conv5_2, 32, 'conv5_3')
        conv5_4 = _conv2d_relu(conv5_3, 34, 'conv5_4')
        avgpool5 = _avgpool(conv5_4)

        return avgpool5

    def make_graph(self, learning_rate):
        with self.graph.as_default():

            # Extract features, a None x 60 x 60 x 512 feature map
            with tf.name_scope("vgg"):
                features = self.load_vgg()

            with tf.name_scope("rpn"):
                # Consider each point in 60 x 60 an anchor
                # For a 3x3 window, there will be 9 anchors
                # Classifications will then be 60 x 60 x 9
                filters1 = tf.get_variable('filters1', shape=(3, 3, 512, 9),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias1 = tf.get_variable('bias1', shape=(9,), initializer=tf.zeros_initializer())
                cls = tf.nn.conv2d(features, filters1, strides=1, padding="SAME")
                cls = tf.nn.bias_add(cls, bias1)
                cls = tf.nn.sigmoid(cls)

                # Each anchor (9) in point indexed by y1, x1, y2, x2
                # 60 x 60 x 36
                filters2 = tf.get_variable('filters2', shape=(3, 3, 512, 36),
                                           initializer=tf.keras.initializers.glorot_normal())
                bias2 = tf.get_variable('bias2', shape=(36,), initializer=tf.zeros_initializer())
                regr = tf.nn.conv2d(features, filters2, strides=1, padding="SAME")
                regr = tf.nn.bias_add(regr, bias2)
                regr = tf.nn.leaky_relu(regr)
                regr = tf.reshape(regr, (60, 60, 9, 4))

            # Calculate ROI
            # Remove boxes out of bounds (just check x, y)
            # NMS with 0.7
            boxes = tf.reshape(regr, (-1, 60*60*9, 4))
            scores = tf.layers.flatten(cls)
            selected_indices = tf.image.non_max_suppression(boxes, scores, iou_threshold=0.7, max_output_size=100)

            init = tf.global_variables_initializer()
        return init, features, selected_indices


# Test program
if __name__ == "__main__":
    rcnn = RCNN()
    waymo_data = WaymoData()
    with tf.Session(graph=rcnn.graph) as sess:
        rcnn.init.run()
        new_batch = waymo_data.get_batch()
        # Batch size 1 for quick testing
        temp_batch = [new_batch[0][0], new_batch[1][0], new_batch[2][0]]
        feed_dict = {rcnn.X: temp_batch[0]}
        # Obtain features
        features, indices = sess.run(rcnn.features, rcnn.indices, feed_dict=feed_dict)
        print(indices)
