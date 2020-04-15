from LoadCity import CityScapes
from ResUNet import ResUNet
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

NUM_EPOCHS = 25000
save_freq = 100
DEBUG = False
learning_rate = 0.007
restore = False
save = True
load_path = "./pretrained/model.ckpt"
batch_size = 32
test_size = 0.01
LEARNING_RATE = 0.007


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

        for epoch in range(NUM_EPOCHS):
            data.shuffle_data()
            avg_loss = 0

            # Mini batches
            while not data.EndOfData:
                batch_x, batch_y = data.get_batch()

                if DEBUG:
                    debug_grads(sess, feed_dict)

                feed_dict = {model.X: batch_x, model.labels: batch_y}
                _, loss_val, outs = sess.run([model.train_op, model.loss, model.predictions], feed_dict=feed_dict)
                avg_loss += loss_val

            cv_x, cv_y = data.get_val_data()
            feed_dict = {model.X: cv_x, model.labels: cv_y}
            cv_loss = sess.run([model.loss], feed_dict=feed_dict)[0]
            print(epoch, "Train Loss", avg_loss, "CV Loss", cv_loss)

            # Save
            if save and (epoch % save_freq == 0):
                save_str = "tmp/model" + str(epoch) + ".ckpt"
                save_path = model.saver.save(sess, save_str)

        # Save final weights
        save_path = model.saver.save(sess, "tmp/model.ckpt")