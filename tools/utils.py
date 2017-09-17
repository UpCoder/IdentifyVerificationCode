# -*- coding=utf-8 -*-
# This file contain the functions that combine with tensorflow
import tensorflow as tf
import numpy as np


'''
    将输入数据整理成tensorflow batch的格式
    :param image_paths 图像的路径
    :param labeles 图像的label
    :param num_epochs 迭代的轮数，默认是None，代表的是不限制轮数，无限次迭代
    :param shuffle　代表的是是否打乱数据
'''
def generate_image_batch_by_filename(dataset,
                                     num_epochs=None,
                                     shuffle=False,
                                     batch_size=128,
                                     image_size=[256, 256],
                                     num_processing_threads=4):
    image_paths = dataset.image_pathes
    labeles_vec = dataset.label_vec
    labeles = np.asarray(labeles_vec, np.float32)
    print np.shape(image_paths)
    print np.shape(labeles)
    filename, label = tf.train.slice_input_producer(
        [image_paths, labeles],
        shuffle=shuffle,
        num_epochs=num_epochs
    )
    print filename
    print label
    num_processing_threads = num_processing_threads
    images_and_labels = []
    for thread_id in range(num_processing_threads):
        image_buff = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_buff, channels=3)

        # It's must be execute the reshape operation, or the shape of image is ?
        image = tf.image.resize_images(image, image_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        print 'image is ', image
        images_and_labels.append([
            image, label
        ])
    print images_and_labels
    batch_images, batch_labels = tf.train.batch_join(
        images_and_labels,
        allow_smaller_final_batch=True,
        batch_size=batch_size
    )
    print batch_labels
    return batch_images, batch_labels
