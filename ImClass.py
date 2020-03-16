from LeNet import LeNet
from cifar10 import Cifar10
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


if __name__ == "__main__":
    # Make graph
    LeNet = LeNet(LEARNING_RATE)
    Cifar10 = Cifar10()

    with tf.Session(graph=LeNet.graph) as sess:
        LeNet.init.run()
        step = 0
        for epoch in range(NUM_EPOCHS):
            Cifar10.reset_pos()
            avg_loss = 0
            training = True
            Cifar10.shuffle_data()
            while Cifar10.pos < Cifar10.N:
                batch_x, batch_y = Cifar10.get_batch(BATCH_SIZE)
                feed_dict = {LeNet.X: batch_x, LeNet.labels: batch_y}

                if Cifar10.pos == -1:
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

            # Accuracy on Test Set
            training = False
            feed_dict = {LeNet.X: Cifar10.train_data, LeNet.labels: Cifar10.train_labels}
            test_outs, test_labs = sess.run([LeNet.predictions, LeNet.labels], feed_dict=feed_dict)
            test_preds = np.argmax(test_outs, axis=1)
            accuracy = 100.0 * np.sum(np.equal(test_labs, test_preds))/len(test_labs)

            print(epoch, avg_loss, accuracy)
            avg_loss = 0
