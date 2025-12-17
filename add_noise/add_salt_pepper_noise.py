import cv2
import random
import os
import numpy as np


def add_salt_pepper(img, prob):
    resultImg = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                resultImg[i][j] = 0
            elif rdn > thres:
                resultImg[i][j] = 255
            else:
                resultImg[i][j] = img[i][j]
    return resultImg


def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        out_img = add_salt_pepper(img, 0.05)
        cv2.waitKey(0)
        cv2.imwrite("../data/ISIC2018(5)_Salt_noise/images" + "/" + filename, out_img)


read_directory("../data/ISIC2018(5)/images")