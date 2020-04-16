import pickle
import pandas as pd
import numpy as np
import glob
import re
import imageio


class Label:
    def __init__(self, name, id, trainId, category, catId, hasInstances, ignoreInEval, color):
        self.catId = catId


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


def id_to_cat(image):
    for row_num in range(len(image)):
        for ind_num in range(len(image[row_num])):
            image[row_num][ind_num] = labels[image[row_num][ind_num]].catId
    return image


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
        batch_y = np.asarray([id_to_cat(imageio.imread(label_extension(file))) for file in batch_files])
        return batch_x, batch_y

    def get_val_data(self, batch_size=2):
        np.random.shuffle(self.val_files)
        val_batch = self.val_files[:batch_size]
        val_x = np.asarray([imageio.imread(data_extension(file, train=False)) for file in val_batch])
        val_y = np.asarray([id_to_cat(imageio.imread(label_extension(file, train=False))) for file in val_batch])
        return val_x, val_y

    def reset_pos(self):
        self.pos = 0


# Test the class
if __name__ == "__main__":
    data = CityScapes()
    for file in data.train_files:
        label_im = imageio.imread(label_extension(file))
        id_to_cat(label_im)
        print(np.unique(label_im))