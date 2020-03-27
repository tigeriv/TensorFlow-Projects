import sys
sys.path.append("waymo_od")

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import numpy as np
import itertools
import glob

tf.enable_eager_execution()


class WaymoData:
    def __init__(self, data_loc='./training_0000', batch_size=8):
        self.fnames = tf.constant([fname for fname in glob.glob(data_loc + '/segment*')])
        self.batch_size = batch_size
        self.dataset = self.make_set()
        self.iter = iter(self.dataset)

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
            return next(self.iter)
        except StopIteration:
            self.make_set()
            return None

    def batch_to_data(self, batch):
        batch_data = []
        for data in batch:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            cam_dict = {}
            for camera in frame.images:
                image = tf.image.decode_jpeg(camera.image)
                cam_dict[camera.name] = [[image.shape], []]
            for camera in frame.projected_lidar_labels:
                for label in camera.labels:
                    label_data = [label.box.center_x, label.box.center_y, label.box.width, label.box.length, label.type]
                    cam_dict[camera.name][1].append(label_data)
            batch_data.append(cam_dict)
        return batch_data

    def show_image(self, image, labels):
        print(image.shape)
        exit()
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for label in labels:
            c_x, c_y, w, h, __ = label
            start_x = int(c_x - (w/2))
            start_y = (c_y - (h/2))
            print(image.shape, c_x, c_y, w, h)
            rect = patches.Rectangle((start_x, start_y), w, h, linewidth=1, edgecolor='r', fill=False)
            ax.add_patch(rect)
        plt.show()


waymo_data = WaymoData()
new_batch = waymo_data.get_batch()
while new_batch is not None:
    waymo_data.batch_to_data(new_batch)
    new_batch = waymo_data.get_batch()
    fr = new_batch[0]
    cam = fr[fr.keys()[0]]
    waymo_data.show_image(cam[0], cam[1])
exit()


data_list = []
label_list = []
fnames = tf.constant([fname for fname in glob.glob('./training_0000/segment*')])
dataset = tf.data.TFRecordDataset(fnames)
dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
dataset = dataset.batch(2, drop_remainder=True)
for batch in dataset:
    for data in batch:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        print(frame.context.name)
exit()

for fname in glob.glob('./training_0000/segment*'):
    dataset = tf.data.TFRecordDataset(fname)
    print(fname)
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # The names of cameras on the car
        print(frame.camera_labels)

        # Image for each camera
        for index, image in enumerate(frame.images):
            im = tf.image.decode_jpeg(image.image)
            print(image.name)
            plt.imshow(im)
            plt.show()

        # 2D Bounding Boxes by camera
        for index, label in enumerate(frame.projected_lidar_labels):
            print(label.name, label)

        # Feature vector of 3D points
        (range_images, camera_projections, range_image_top_pose) = (
            frame_utils.parse_range_image_and_camera_projection(frame))
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections,
                                                                           range_image_top_pose)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(frame, range_images,
                                                                                   camera_projections,
                                                                                   range_image_top_pose, ri_index=1)
        points_all = np.concatenate(points, axis=0)
        data_list.append(points_all)
        label_list.append(labels)
np.savez('PC3D.bin', *data_list)
npzfile = np.load('PC3D.bin.npz')
print(npzfile.files)
np.savez('Label3D.bin', *label_list)
npzfile = np.load('Label3D.bin.npz')
print(npzfile.files)
