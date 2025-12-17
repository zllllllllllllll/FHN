import cv2
import os
import numpy as np


def add_noise_Guass(img, mean=0, var=0.01):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out_img = img + noise
    if out_img.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
        out_img = np.clip(out_img, low_clip, 1.0)
        out_img = np.uint8(out_img * 255)
    return out_img


def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        out_img = add_noise_Guass(img)
        # cv2.imshow("img", out_img)
        cv2.waitKey(0)
        cv2.imwrite("../data/ISIC2018(5)_Gauss_noise/images" + "/" + filename, out_img * 255)


read_directory("../data/ISIC2018(5)/images")
