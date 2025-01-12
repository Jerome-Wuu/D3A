# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
from PIL import Image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image



def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new(mode='RGB', size=size, color=(128, 128, 128))
    new_image.paste(image, ((w-nw) // 2, (h - nh) // 2))
    return new_image, nw, nh


def preprocess_input(image):
    image /= 255.0
    return image