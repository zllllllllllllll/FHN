import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决中文乱码等问题


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


trn_binary = np.load("./model_chestX-ray/mymodel/database_binary_chestX-ray.npy")
trn_label = np.load("./model_chestX-ray/mymodel/database_label_chestX-ray.npy")
tst_binary = np.load("./model_chestX-ray/mymodel/test_binary_chestX-ray.npy")
tst_label = np.load("./model_chestX-ray/mymodel/test_label_chestX-ray.npy")
print('测试集的大小：', tst_binary.shape)
img_dir = "../data/chestX-ray/"
# img_dir = "../data/ISIC2018(5)/"
with open("../data/chestX-ray/database_demo.txt", "r") as f:
    trn_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]
# with open("../data/chestX-ray/test_demo.txt", "r") as f:
# with open("../data/ISIC2018_Enhance/test_demo.txt", "r") as f:
with open("../data/chestX-ray/test_demo.txt", "r") as f:
    tst_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]

m = 7
n = 10
plt.figure(figsize=(40, 20), dpi=50)
font_size = 25
# tst_select_index = np.random.permutation(range(tst_binary.shape[0]))[0: m]  # 从测试集中随机挑选m个样本进行查询
tst_select_index = [15, 219, 31, 102, 369, 274, 340]
# tst_select_index = [426, 464, 476, 485, 421]
# names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL']
print('从测试集中随机挑选m个样本的索引：', tst_select_index)

for row, query_index in enumerate(tst_select_index):
    query_binary = tst_binary[query_index]
    query_label = tst_label[query_index]
    # 计算测试集和检索是否相似
    gnd = (np.dot(query_label, trn_label.transpose()) > 0).astype(np.float32)
    # 通过哈希码计算汉明距离
    hamm = CalcHammingDist(query_binary, trn_binary)
    # 计算最近的n个距离的索引
    ind = np.argsort(hamm)[:n]
    print('数据库样本的索引：', ind)
    # 返回结果的真值
    t_gnd = gnd[ind]
    # 返回结果的汉明距离
    q_hamm = hamm[ind].astype(int)
    q_img_path = tst_img_path[query_index]
    return_img_list = np.array(trn_img_path)[ind].tolist()
    plt.subplot(m, n + 1, row * (n+1) + 1)
    img = Image.open(q_img_path).convert('RGB').resize((128, 128))
    plt.imshow(img)
    plt.axis('off')
    plt.text(30, 145, '查询样本', size=font_size)
    for index, img_path in enumerate(return_img_list):
        # plt.subplot(1, n + 1, index + 2)
        plt.subplot(m, n + 1, row * (n+1) + index + 2)
        img = Image.open(img_path).convert('RGB').resize((120, 120))
        if t_gnd[index]:
            plt.text(60, 145, '√', size=font_size)
            img = ImageOps.expand(img, 4, fill=(0, 0, 255))  # 蓝色
        else:
            plt.text(60, 145, '×', size=font_size)
            img = ImageOps.expand(img, 4, fill=(255, 0, 0))  # 红色
        plt.axis('off')
        plt.imshow(img)
plt.savefig("demo.png", dpi=200)
plt.show()
