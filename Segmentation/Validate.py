from LoadCity import CityScapes
from ResUNet import ResUNet
from LoadCity import *
from FastSCNN import FSCNN
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt


load_path = "./tmp/model2.ckpt"


def load_sess(path, model):
    sess = tf.Session(graph=model.graph)
    model.saver.restore(sess, path)
    return sess


def predict(sess, model, images):
    feed_dict = {model.X: images}
    outs = sess.run([model.predictions], feed_dict=feed_dict)[0]
    sess_temp = tf.Session()
    with sess_temp.as_default():
        labels = tf.nn.softmax(outs).eval(session=sess_temp)
    return labels


def display_image(image):
    plt.imshow(image)
    plt.show()


def display_label_image(image):
    image = cat_to_im(image)
    display_image(image)


if __name__ == "__main__":
    model = ResUNet(width=2048, height=1024)
    data = CityScapes()
    sess = load_sess(load_path, model)
    while True:
        val_x, val_y = data.get_val_data(batch_size=1)
        val_labels = predict(sess, model, val_x)
        for index in range(len(val_labels)):
            display_image(val_x[index])
            cat_image = np.argmax(val_labels[index], axis=-1)
            display_label_image(cat_image)