from LoadCity import CityScapes
from LoadCity import *
from ResUNet import ResUNet
from FastSCNN import FSCNN
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Training settings
NUM_EPOCHS = 100
save_freq = 1
restore = False
save = True
load_path = "./pretrained/model.ckpt"
batch_size = 2
test_size = 0.01
LEARNING_RATE = 0.07


# Debugging settings
DEBUG = False
TIME_PREDICTION = True
TIME_INTERVAL = True


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
    model = ResUNet(LEARNING_RATE, width=2048, height=1024)
    data = CityScapes()

    with tf.Session(graph=model.graph) as sess:
        if restore:
            model.saver.restore(sess, load_path)
            NUM_EPOCHS = 0
        else:
            model.init.run()

        iteration = 0
        start = time.time()
        for epoch in range(NUM_EPOCHS):
            data.shuffle_data()
            avg_loss = 0

            # Mini batches
            while not data.EndOfData:
                batch_x, batch_y = data.get_batch(batch_size)
                batch_x, batch_y = augment_batch(batch_x, batch_y)
                model.width = batch_x.shape[2]
                model.height = batch_x.shape[1]
                feed_dict = {model.X: batch_x, model.labels: batch_y}

                if DEBUG:
                    debug_grads(sess, model, feed_dict)

                if TIME_PREDICTION:
                    start = time.time()
                    outs = sess.run([model.predictions], feed_dict=feed_dict)
                    end = time.time()
                    print("Prediction", end - start)

                _, loss_val, outs = sess.run([model.train_op, model.loss, model.predictions], feed_dict=feed_dict)
                avg_loss += loss_val
                end = time.time()
                iteration += 1
                if iteration % 100 == 0 and TIME_INTERVAL:
                    print("Iteration", iteration, " took", end-start, " seconds")
                    start = time.time()

            cv_x, cv_y = data.get_val_data(batch_size)
            # Make it a smaller amount of data so we don't crash :)
            feed_dict = {model.X: cv_x, model.labels: cv_y}
            cv_loss = sess.run([model.loss], feed_dict=feed_dict)[0]
            print(epoch, "Train Loss", avg_loss, "CV Loss", cv_loss)

            # Save
            if save and (epoch % save_freq == 0):
                save_str = "tmp/model" + str(epoch) + ".ckpt"
                save_path = model.saver.save(sess, save_str)

        # Save final weights
        save_path = model.saver.save(sess, "tmp/model.ckpt")