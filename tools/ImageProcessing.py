# -*- coding=utf-8 -*-
import os
from PIL import Image
import numpy as np


'''
    将指定的图像转化为灰度图像
    :param image_path 原图像的路径
    :param save_path　将要存储的路径
'''
def convert_gray(image_path, save_path=None):
    image = Image.open(image_path)
    image = image.convert('L')

    if save_path is not None:
        image.save(save_path)
    else:
        image.show()


'''
    将文件夹下面的所有图像都转变为灰度图像
    :param image_dir 存放图片的文件夹
    :param save_dir　转化成灰度图像之后的存放的文件夹
'''
def convert_images_gray(image_dir, save_dir):
    image_names = os.listdir(image_dir)
    image_pathes = [os.path.join(image_dir, image_name) for image_name in image_names]
    save_pathes = [os.path.join(save_dir, image_name) for image_name in image_names]

    for index, image_path in enumerate(image_pathes):
        print image_path
        convert_gray(image_path, save_pathes[index])

'''
    将一段文本转化为向量,不区分大小写，所以每个字母的维度是10+26
    :param txt文本
    :return 对应的向量
'''
def txt2vec(txt):
    # the max length of singel vector
    MAX_SINGLE_LENGTH = 36
    vec = np.zeros(MAX_SINGLE_LENGTH*len(txt), np.uint8)

    for index, c in enumerate(txt):
        single_vec = np.zeros(MAX_SINGLE_LENGTH, np.uint8)
        if '0' <= c <= '9':
            pos = ord(c)-ord('0')
            single_vec[pos] = 1
        else:
            c = (''+c).lower()
            pos = ord(c) - ord('a') + 10
            single_vec[pos] = 1
        vec[index*MAX_SINGLE_LENGTH: (index+1)*MAX_SINGLE_LENGTH] = single_vec
    return vec

'''
    将上述生成的ｖｅｃ转化为十进制的数
'''
def vec2num(vec, vc_len = 4, letter_len=36):
    num = 0
    for i in range(vc_len):
        print np.argmax(vec[i*letter_len:(i+1)*letter_len])*((4)**(vc_len-i-1))
        num += np.argmax(vec[i*letter_len:(i+1)*letter_len])*((4)**(vc_len-i-1))
    return num
'''
    将一个label转化为txt
    :param label 对应的ｌａｂｅｌ值
    :return 返回我们的文本
'''

if __name__ == '__main__':
    # convert_images_gray(
    #     '/home/give/Documents/dataset/VerficationCode/Captcha/test_images',
    #     '/home/give/Documents/dataset/VerficationCode/Captcha/test_images_gray')
    print vec2num(txt2vec('zzzz'))
