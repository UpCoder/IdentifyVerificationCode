import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
from resnet_train import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from resnet import inference

from synset import *
from synset import *
from DataSet.DataSetBridger import generate_batch_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames


def load_data(data_dir):
    data = []
    i = 0

    print "listing files in", data_dir
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    print "took %f sec" % duration

    for img_fn in files:
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = re.search(r'(n\d+)', img_fn).group(1)
        fn = os.path.join(data_dir, img_fn)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

    return data


def distorted_inputs():
    train_image_dir = '/home/give/Documents/dataset/VerficationCode/Captcha/captcha_train_gray'
    val_image_dir = '/home/give/Documents/dataset/VerficationCode/Captcha/captcha_val_gray'
    train_image_batch, train_label_batch = generate_batch_data(train_image_dir, batch_size=FLAGS.batch_size)
    val_image_batch, val_label_batch = generate_batch_data(val_image_dir, batch_size=FLAGS.batch_size)
    return [train_image_batch, train_label_batch], [val_image_batch, val_label_batch]


def main(_):
    [train_images, train_labels], [val_images, val_labels] = distorted_inputs()
    is_training = tf.placeholder('bool', [], name='is_training')
    images, labels = tf.cond(is_training,
                             lambda: (train_images, train_labels),
                             lambda: (val_images, val_labels))
    logits_multi_task = []
    for i in range(FLAGS.letter_num_per_vc):
        logits = inference(images,
                           task_name='task_' + str(i),
                           num_classes=FLAGS.max_single_vc_length,
                           is_training=True,
                           bottleneck=False,)
        logits_multi_task.append(logits)
    save_model_path = '/home/give/PycharmProjects/AIChallenger/ResNet/models'
    train(is_training, logits_multi_task, images, labels, save_model_path=save_model_path)


if __name__ == '__main__':
    tf.app.run()
