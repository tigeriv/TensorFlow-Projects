import sys
sys.path.append("waymo_od")

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import glob
import cv2

tf.enable_eager_execution()


# Size is (height, width)
class WaymoData:
    def __init__(self, data_loc='./training_0000', batch_size=8, image_size=(1920, 1920)):
        self.fnames = tf.constant([fname for fname in glob.glob(data_loc + '/segment*')])
        self.batch_size = batch_size
        self.dataset = self.make_set()
        self.iter = iter(self.dataset)
        self.image_size = image_size

    def make_set(self):
        dataset = tf.data.TFRecordDataset(self.fnames)
        dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def get_set(self):
        return self.dataset

    # Return None and create new DataSet if no more batches
    def get_batch(self):
        try:
            new_batch = next(self.iter)
            batch_data = self.batch_to_data(new_batch)
            self.make_images_uniform(batch_data)
            return batch_data
        except StopIteration:
            self.make_set()
            return None

    # Data is a list (batch) of dicts
    # Dict keys are cameras
    # Data in dict is a list of [image, [boxes], [types]]
    # Label data actually goes [center_x, center_y, height, width] of box
    def batch_to_data(self, batch):
        batch_data = []
        for data in batch:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            cam_dict = {}
            for camera in frame.images:
                image = (tf.image.decode_jpeg(camera.image)).numpy()
                cam_dict[camera.name] = [image, [], []]
            for camera in frame.projected_lidar_labels:
                for label in camera.labels:
                    label_data = [label.box.center_x, label.box.center_y, label.box.width, label.box.length]
                    cam_dict[camera.name][1].append(label_data)
                    cam_dict[camera.name][2].append(label.type)
                cam_dict[camera.name][1] = np.array(cam_dict[camera.name][1])
            batch_data.append(cam_dict)
        return batch_data

    # Note that while most data is good, some is not
    def show_image(self, image, labels):
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for label in labels:
            c_x, c_y, h, w = label
            start_x = int(c_x - (w/2))
            start_y = (c_y - (h/2))
            rect = patches.Rectangle((start_x, start_y), w, h, linewidth=1, edgecolor='r', fill=False)
            ax.add_patch(rect)
        plt.show()

    # Adjusts image to the size the class is set to handle
    # Also changes bounding boxes
    def make_images_uniform(self, batch):
        for frame in batch:
            for camera_name in frame.keys():
                image, labels, types = frame[camera_name]
                new_image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                height_scale = new_image.shape[0]/image.shape[0]
                width_scale = new_image.shape[1]/image.shape[1]
                try:
                    labels *= [width_scale, height_scale, height_scale, width_scale]
                except ValueError:
                    # No boxes
                    pass
                frame[camera_name] = [new_image, labels, types]


# Test program
if __name__ == "__main__":
    waymo_data = WaymoData()
    new_batch = waymo_data.get_batch()
    while new_batch is not None:
        for frame in new_batch:
            for image, labels, types in frame.values():
                waymo_data.show_image(image, labels)
        new_batch = waymo_data.get_batch()