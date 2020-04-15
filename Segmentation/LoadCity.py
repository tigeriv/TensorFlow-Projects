import pickle
import pandas as pd
import numpy as np
import glob
import re
import imageio
import cityscapesscripts


def extract_file_name(file):
    return re.search(r'\w+/\w+_\d+_\d+', file).group(0)


def data_extension(file, train=True):
    if train:
        return './leftImg8bit_trainvaltest/leftImg8bit/train/' + file + '_leftImg8bit.png'
    else:
        return './leftImg8bit_trainvaltest/leftImg8bit/val/' + file + '_leftImg8bit.png'


# Class id's, currently 33 of them
def label_extension(file, train=True):
    if train:
        return './gtFine_trainvaltest/gtFine/train/' + file + '_gtFine_labelIds.png'
    else:
        return './gtFine_trainvaltest/gtFine/val/' + file + '_gtFine_labelIds.png'


# Same as label, except if there are multiple of the class, it is multiplied by a thousand
# And remainder is the instance number
def instance_extension(file, train=True):
    if train:
        return './gtFine_trainvaltest/gtFine/train/' + file + '_gtFine_instanceIds.png'
    else:
        return './gtFine_trainvaltest/gtFine/val/' + file + '_gtFine_instanceIds.png'


class CityScapes:
    def __init__(self):
        self.train_files, self.val_files = self.get_files()
        self.N = len(self.train_files)
        self.pos = 0
        self.EndOfData = False

    def get_files(self):
        # Obtain train and test data
        train_files = []
        val_files = []
        for place in glob.glob('./leftImg8bit_trainvaltest/leftImg8bit/train/*'):
            train_files += [extract_file_name(file) for file in glob.glob(place + "/*")]
        for place in glob.glob('./leftImg8bit_trainvaltest/leftImg8bit/val/*'):
            val_files += [extract_file_name(file) for file in glob.glob(place + "/*")]
        return train_files, val_files

    def shuffle_data(self):
        self.pos = 0
        np.random.shuffle(self.train_files)
        self.EndOfData = False

    def get_batch(self, batch_size=2):
        batch_files = self.train_files[self.pos: self.pos+batch_size]
        self.pos += batch_size
        if self.pos >= len(self.train_files):
            self.EndOfData = True
        batch_x = np.asarray([imageio.imread(data_extension(file)) for file in batch_files])
        batch_y = np.asarray([imageio.imread(label_extension(file)) for file in batch_files])
        return batch_x, batch_y

    def get_val_data(self):
        val_x = np.asarray([imageio.imread(data_extension(file)) for file in self.val_files])
        val_y = np.asarray([imageio.imread(label_extension(file)) for file in self.val_files])
        return val_x, val_y

    def reset_pos(self):
        self.pos = 0


# Test the class
if __name__ == "__main__":
    data = CityScapes()
    for file in data.train_files:
        label_im = imageio.imread(label_extension(file))
        print(np.unique(label_im))