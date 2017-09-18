# -*- coding=utf-8 -*-
from DataSet import DataSetBase as DataSet
from tools.utils import generate_image_batch_by_filename
import tensorflow as tf
import numpy as np

'''
    生成深度学习需要用的一个ｂａｔｃｈ的数据
    :param image_dir 输入的图像文件夹
    :return two tensors, one is the batch data of images, one is the batch data of labels
'''
def generate_batch_data(image_dir,
                        num_epochs=None,
                        shuffle=False,
                        batch_size=128,
                        image_size=[256, 256],
                        num_processing_threads=4):
    dataset = DataSet(image_dir)
    return generate_image_batch_by_filename(
        dataset,
        num_epochs,
        shuffle,
        batch_size,
        image_size,
        num_processing_threads
    )

if __name__ == '__main__':
    image_batch, label_batch = generate_batch_data(
        '/home/give/Documents/dataset/VerficationCode/Captcha/captcha_val_gray',
        num_epochs=1
    )
    print image_batch
    print label_batch
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while True:
        try:
            batch_label = sess.run(label_batch)
            print batch_label[0]
            print np.shape(sess.run(label_batch))
            # print sess.run(label_batch)
        except Exception, error:
            print 'finish', error.message
            break
