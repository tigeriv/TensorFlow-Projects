import pickle
import pandas as pd
import numpy as np


class Cifar10:
    def __init__(self):
        self.train_data, self.train_labels, self.test_data, self.test_labels = self.get_data()
        self.N = len(self.train_labels)
        self.pos = 0
        self.indices = np.arange(self.N)

    def get_data(self):
        # Obtain train and test data
        train_files = ["cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/data_batch_2",
                       "cifar-10-batches-py/data_batch_3", "cifar-10-batches-py/data_batch_4",
                       "cifar-10-batches-py/data_batch_5"]
        train_labels, train_data = [], -1
        first_time = True
        for file in train_files:
            train_dict = self.unpickle(file)
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

        test_dict = self.unpickle("cifar-10-batches-py/test_batch")
        _, test_labels, test_data, _ = test_dict[b"batch_label"], test_dict[b"labels"], test_dict[b"data"], \
                                test_dict[b"filenames"]

        test_x = np.reshape(test_data, (len(test_labels), 32, 32, 3))
        test_y = test_labels
        return train_data, train_labels, test_x, test_y

    # Returns a dictionary, where data = numpy uint8 array 10000x3072 (RGB 32x32 images)
    # labels = list of 10000 numbers in range 0-9
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def shuffle_data(self):
        np.random.shuffle(self.indices)

    def get_batch(self, batch_size):
        batch_indices = self.indices[self.pos: self.pos+batch_size]
        batch_x = self.train_data[batch_indices]
        batch_y = [self.train_labels[i] for i in batch_indices]
        self.pos += batch_size
        return batch_x, batch_y

    def reset_pos(self):
        self.pos = 0