# -*- coding=utf-8 -*-
import os

from tools.ImageProcessing import txt2vec


class DataSetBase:
    def __init__(self, image_dir):
        self.image_names = os.listdir(image_dir)
        self.image_pathes = [os.path.join(image_dir, image_name) for image_name in self.image_names]
        self.label_str = [image_name.split('.')[0] for image_name in self.image_names]
        # convert the text to the vector.
        self.label_vec = [txt2vec(label_str) for label_str in self.label_str]
