import cv2
import os
import numpy as np


def random_noise(image, noise_num):
    img_noise = image
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        out_img = random_noise(img, 10000)
        # cv2.imshow("img", out_img)
        cv2.waitKey(0)
        cv2.imwrite("../data/ISIC2018(5)_random_noise/images" + "/" + filename, out_img)


read_directory("../data/chestX-ray/images")