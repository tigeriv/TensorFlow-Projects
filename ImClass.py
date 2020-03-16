from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import pickle
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

BATCH_SIZE = 1024
NUM_EPOCHS = 500
LEARNING_RATE = 0.0007
training = True


# Returns a dictionary, where data = numpy uint8 array 10000x3072 (RGB 32x32 images)
# labels = list of 10000 numbers in range 0-9
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == "__main__":
    # Obtain train and test data
    train_files = ["cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/data_batch_2", "cifar-10-batches-py/data_batch_3"]
    train_labels, train_data = [], -1
    first_time = True
    for file in train_files:
        train_dict = unpickle(file)
        _, temp_labels, temp_data, _ = train_dict[b"batch_label"], train_dict[b"labels"], train_dict[b"data"], \
                                         train_dict[b"filenames"]
        temp_labels, temp_data = temp_labels, temp_data
        train_labels += temp_labels
        if first_time:
            train_data = temp_data
            first_time = False
        else:
            train_data = np.concatenate([train_data, temp_data], axis=0)
    train_data = np.reshape(train_data, (len(train_labels), 32, 32, 3))

    test_dict = unpickle("cifar-10-batches-py/test_batch")
    _, test_labels, test_data, _ = test_dict[b"batch_label"], test_dict[b"labels"], test_dict[b"data"], \
                            test_dict[b"filenames"]

    test_x = np.reshape(test_data, (len(test_labels), 32, 32, 3))
    test_y = test_labels


    # Make graph
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, (None, 32, 32, 3))
        labels = tf.placeholder(tf.int32, (None,))
        # filter shape: filter_height, filter_width, in_channels, out_channels
        with tf.name_scope("conv1"):
            filters1 = tf.get_variable('filters1', shape=(5, 5, 3, 6), initializer=tf.keras.initializers.glorot_normal())
            bias1 = tf.get_variable('bias1', shape=(6,), initializer=tf.zeros_initializer())
            X1 = tf.nn.conv2d(X, filters1, strides=1, padding="VALID")
            X1 = tf.nn.bias_add(X1, bias1)
            X1 = tf.nn.leaky_relu(X1)
            X1 = tf.nn.avg_pool2d(X1, ksize=2, strides=2, padding="VALID")
        with tf.name_scope("conv2"):
            filters2 = tf.get_variable('filters2', shape=(5, 5, 6, 16), initializer=tf.keras.initializers.glorot_normal())
            bias2 = tf.get_variable('bias2', shape=(16,), initializer=tf.zeros_initializer())
            X2 = tf.nn.conv2d(X1, filters2, strides=1, padding="VALID")
            X2 = tf.nn.bias_add(X2, bias2)
            X2 = tf.nn.leaky_relu(X2)
            X2 = tf.nn.avg_pool2d(X2, ksize=2, strides=2, padding="VALID")
            X2 = tf.layers.flatten(X2)
        with tf.name_scope("fcn1"):
            W1 = tf.get_variable('W1', shape=(400, 120), initializer=tf.keras.initializers.glorot_normal())
            b1 = tf.Variable(tf.zeros((120,)), trainable=True)
            X3 = tf.nn.leaky_relu(tf.matmul(X2, W1) + b1)
        with tf.name_scope("fcn2"):
            W2 = tf.get_variable('W2', shape=(120, 84), initializer=tf.keras.initializers.glorot_normal())
            b2 = tf.Variable(tf.zeros((84,)), trainable=True)
            X4 = tf.nn.leaky_relu(tf.matmul(X3, W2) + b2)
        with tf.name_scope("softmax"):
            W3 = tf.get_variable('W3', shape=(84, 10), initializer=tf.keras.initializers.glorot_normal())
            b3 = tf.Variable(tf.zeros((10,)), trainable=True)
            X5 = tf.add(tf.matmul(X4, W3), b3)
            if training:
                X5 = tf.nn.dropout(X5, rate=0.2)

        predictions = tf.nn.softmax(X5)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
        loss = tf.reduce_mean(cost)

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        step = 0
        for epoch in range(NUM_EPOCHS):
            pos = 0
            avg_loss = 0
            training = True
            while pos < len(train_labels):
                try:
                    batch_x = train_data[pos:pos + BATCH_SIZE]
                    batch_y = train_labels[pos:pos+BATCH_SIZE]
                    feed_dict = {X: batch_x, labels: batch_y}
                except:
                    pos = 10000
                    continue

                if pos == -1:
                    var_list = (variables.trainable_variables() + ops.get_collection(
                        ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
                    print('variables')
                    for v in var_list:
                        print('  ', v.name)
                    # get all gradients
                    grads_and_vars = optimizer.compute_gradients(loss)
                    train_op = optimizer.apply_gradients(grads_and_vars)

                    zipped_val = sess.run(grads_and_vars, feed_dict=feed_dict)

                    for rsl, tensor in zip(zipped_val, grads_and_vars):
                        print('-----------------------------------------')
                        print('name', tensor[0].name.replace('/tuple/control_dependency_1:0', '').replace('gradients/', ''))
                        print('gradient', rsl[0])
                        print('value', rsl[1])
                    exit()

                _, loss_val, outs, labs, W_check = sess.run([train_op, loss, predictions, labels, W3], feed_dict=feed_dict)
                avg_loss += loss_val
                step += 1
                pos += BATCH_SIZE

            # Accuracy on Test Set
            training = False
            feed_dict = {X: test_x, labels: test_y}
            test_outs, test_labs = sess.run([predictions, labels], feed_dict=feed_dict)
            test_preds = np.argmax(test_outs, axis=1)
            accuracy = 100.0 * np.sum(np.equal(test_labs, test_preds))/len(test_labs)

            print(epoch, avg_loss, accuracy)
            avg_loss = 0
