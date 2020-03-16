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
DEBUG = False


def debug_grads(sess, model, feed_dict):
    var_list = (variables.trainable_variables() + ops.get_collection(
        ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    print('variables')
    for v in var_list:
        print('  ', v.name)
    # get all gradients
    grads_and_vars = model.optimizer.compute_gradients(model.loss)
    train_op = model.optimizer.apply_gradients(grads_and_vars)

    zipped_val = sess.run(grads_and_vars, feed_dict=feed_dict)

    for rsl, tensor in zip(zipped_val, grads_and_vars):
        print('-----------------------------------------')
        print('name', tensor[0].name.replace('/tuple/control_dependency_1:0', '').replace('gradients/', ''))
        print('gradient', rsl[0])
        print('value', rsl[1])


if __name__ == "__main__":
    model = LeNet(LEARNING_RATE)
    data = Cifar10()

    with tf.Session(graph=model.graph) as sess:
        model.init.run()

        for epoch in range(NUM_EPOCHS):
            data.reset_pos()
            avg_loss = 0
            model.Training = True
            data.shuffle_data()

            while data.pos < data.N:
                batch_x, batch_y = data.get_batch(BATCH_SIZE)
                feed_dict = {model.X: batch_x, model.labels: batch_y}
                if DEBUG:
                    debug_grads(sess, model, feed_dict)
                _, loss_val, outs, labs = sess.run([model.train_op, model.loss, model.predictions, model.labels], feed_dict=feed_dict)
                avg_loss += loss_val

            # Accuracy on Test Set
            model.Training = False
            feed_dict = {model.X: data.train_data, model.labels: data.train_labels}
            test_outs, test_labs = sess.run([model.predictions, model.labels], feed_dict=feed_dict)
            test_preds = np.argmax(test_outs, axis=1)
            accuracy = 100.0 * np.sum(np.equal(test_labs, test_preds))/len(test_labs)

            print("Epoch:", epoch, "Average loss:", avg_loss, "Test accuracy:", accuracy)
