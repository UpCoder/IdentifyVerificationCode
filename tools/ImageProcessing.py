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
    将一个vec转化为txt
    :param vec 对应的vec值
    :param vc_len 代表的是一个验证码的长度
    :param letter_len 代表的是每个验证码字符的组成
    :return 返回我们的文本
'''
def vec2txt(vec, vc_len=4, letter_len=36):
    txt = ''
    for i in range(vc_len):
        start = i * letter_len
        end = (i+1) * letter_len
        word = vec[start:end]
        one_index = np.argmax(word)
        if one_index <= 9:
            txt += str(one_index)
        else:
            txt += chr(ord('a') + (one_index-10))
    return txt


'''
    将图片转移到指令目录，并且用指定的名字
    :param result_dict 结果的字典类型的数据，key是文件名，value是预测的验证码值
    :param org_dir 存放图像的文件夹
    :param target_dir 预测结果的文件夹
    
'''
def copy_result_to(result_dict, org_dir, target_dir):
    for key in result_dict.keys():
        image_path = os.path.join(org_dir, key)
        image = Image.open(image_path)
        image.save(os.path.join(target_dir, result_dict[key]+'.jpg'))

if __name__ == '__main__':
    convert_images_gray(
        '/home/give/Documents/dataset/VerficationCode/Captcha/ new_dataset/data',
        '/home/give/Documents/dataset/VerficationCode/Captcha/ new_dataset/data')
    # print vec2txt(txt2vec('zzzz'))
