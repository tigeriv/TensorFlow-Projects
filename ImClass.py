from LeNet import LeNet
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import pickle
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

BATCH_SIZE = 128
NUM_EPOCHS = 500
LEARNING_RATE = 0.007
training = True


# Returns a dictionary, where data = numpy uint8 array 10000x3072 (RGB 32x32 images)
# labels = list of 10000 numbers in range 0-9
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == "__main__":
    # Obtain train and test data
    train_files = ["cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/data_batch_2",
                   "cifar-10-batches-py/data_batch_3", "cifar-10-batches-py/data_batch_4",
                   "cifar-10-batches-py/data_batch_5"]
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
    LeNet = LeNet(LEARNING_RATE)

    with tf.Session(graph=LeNet.graph) as sess:
        LeNet.init.run()
        step = 0
        for epoch in range(NUM_EPOCHS):
            pos = 0
            avg_loss = 0
            training = True
            # Randomly select indices
            rand_indices = np.arange(len(train_labels))
            while pos < len(train_labels):
                batch_indices = rand_indices[pos:pos+BATCH_SIZE]
                batch_x = train_data[batch_indices]
                batch_y = [train_labels[i] for i in batch_indices]
                feed_dict = {LeNet.X: batch_x, LeNet.labels: batch_y}

                if pos == -1:
                    var_list = (variables.trainable_variables() + ops.get_collection(
                        ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
                    print('variables')
                    for v in var_list:
                        print('  ', v.name)
                    # get all gradients
                    grads_and_vars = LeNet.optimizer.compute_gradients(LeNet.loss)
                    train_op = LeNet.optimizer.apply_gradients(grads_and_vars)

                    zipped_val = sess.run(grads_and_vars, feed_dict=feed_dict)

                    for rsl, tensor in zip(zipped_val, grads_and_vars):
                        print('-----------------------------------------')
                        print('name', tensor[0].name.replace('/tuple/control_dependency_1:0', '').replace('gradients/', ''))
                        print('gradient', rsl[0])
                        print('value', rsl[1])
                    exit()

                _, loss_val, outs, labs = sess.run([LeNet.train_op, LeNet.loss, LeNet.predictions, LeNet.labels], feed_dict=feed_dict)
                avg_loss += loss_val
                step += 1
                pos += BATCH_SIZE

            # Accuracy on Test Set
            training = False
            feed_dict = {LeNet.X: test_x, LeNet.labels: test_y}
            test_outs, test_labs = sess.run([LeNet.predictions, LeNet.labels], feed_dict=feed_dict)
            test_preds = np.argmax(test_outs, axis=1)
            accuracy = 100.0 * np.sum(np.equal(test_labs, test_preds))/len(test_labs)

            print(epoch, avg_loss, accuracy)
            avg_loss = 0
