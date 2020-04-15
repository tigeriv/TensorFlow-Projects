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

    def get_batch(self, batch_size=32):
        batch_files = self.train_files[self.pos: self.pos+batch_size]
        self.pos += batch_size
        if self.pos >= len(self.train_files):
            self.EndOfData = True
        batch_x = [imageio.imread(data_extension(file)) for file in batch_files]
        batch_y = [imageio.imread(label_extension(file)) for file in batch_files]
        return batch_x, batch_y

    def get_val_data(self):
        val_x = [imageio.imread(data_extension(file)) for file in self.val_files]
        val_y = [imageio.imread(label_extension(file)) for file in self.val_files]
        return val_x, val_y

    def reset_pos(self):
        self.pos = 0


# Test the class
if __name__ == "__main__":
    data = CityScapes()
    sample_file = data.train_files[0]

    data_file = data_extension(sample_file)
    label_file = label_extension(sample_file)
    instance_file = instance_extension(sample_file)

    data_im = imageio.imread(data_file)
    label_im = imageio.imread(label_file)
    inst_im = imageio.imread(instance_file)

    print(np.unique(inst_im))