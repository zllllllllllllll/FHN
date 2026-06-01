import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})


trn_binary = np.load("./model_ISIC2018(5)/mymodel/trn_binary.npy")
trn_label = np.load("./model_ISIC2018(5)/mymodel/trn_label.npy")
tst_binary = np.load("./model_ISIC2018(5)/mymodel/tst_binary.npy")
tst_label = np.load("./model_ISIC2018(5)/mymodel/tst_label.npy")

trn_binary_DBDH = np.load("./model_ISIC2018(5)/DBDH/trn_binary.npy")
trn_label_DBDH = np.load("./model_ISIC2018(5)/DBDH/trn_label.npy")
tst_binary_DBDH = np.load("./model_ISIC2018(5)/DBDH/tst_binary.npy")
tst_label_DBDH = np.load("./model_ISIC2018(5)/DBDH/tst_label.npy")

trn_binary_DPSH = np.load("./model_ISIC2018(5)/DPSH/trn_binary.npy")
trn_label_DPSH = np.load("./model_ISIC2018(5)/DPSH/trn_label.npy")
tst_binary_DPSH = np.load("./model_ISIC2018(5)/DPSH/tst_binary.npy")
tst_label_DPSH = np.load("./model_ISIC2018(5)/DPSH/tst_label.npy")

trn_binary_IDHN = np.load("./model_ISIC2018(5)/IDHN/trn_binary.npy")
trn_label_IDHN = np.load("./model_ISIC2018(5)/IDHN/trn_label.npy")
tst_binary_IDHN = np.load("./model_ISIC2018(5)/IDHN/tst_binary.npy")
tst_label_IDHN = np.load("./model_ISIC2018(5)/IDHN/tst_label.npy")

trn_binary_VTS = np.load("./model_ISIC2018(5)/VTS/trn_binary.npy")
trn_label_VTS = np.load("./model_ISIC2018(5)/VTS/trn_label.npy")
tst_binary_VTS = np.load("./model_ISIC2018(5)/VTS/tst_binary.npy")
tst_label_VTS = np.load("./model_ISIC2018(5)/VTS/tst_label.npy")

trn_binary_SADH = np.load("./model_ISIC2018(5)/SADH/trn_binary.npy")
trn_label_SADH = np.load("./model_ISIC2018(5)/SADH/trn_label.npy")
tst_binary_SADH = np.load("./model_ISIC2018(5)/SADH/tst_binary.npy")
tst_label_SADH = np.load("./model_ISIC2018(5)/SADH/tst_label.npy")

trn_binary_DPN = np.load("./model_ISIC2018(5)/DPN/trn_binary.npy")
trn_label_DPN = np.load("./model_ISIC2018(5)/DPN/trn_label.npy")
tst_binary_DPN = np.load("./model_ISIC2018(5)/DPN/tst_binary.npy")
tst_label_DPN = np.load("./model_ISIC2018(5)/DPN/tst_label.npy")

trn_binary_HybridHash = np.load("./model_ISIC2018(5)/HybridHash/trn_binary.npy")
trn_label_HybridHash = np.load("./model_ISIC2018(5)/HybridHash/trn_label.npy")
tst_binary_HybridHash = np.load("./model_ISIC2018(5)/HybridHash/tst_binary.npy")
tst_label_HybridHash = np.load("./model_ISIC2018(5)/HybridHash/tst_label.npy")

trn_binary_DGSSH = np.load("./model_ISIC2018(5)/DGSSH/trn_binary.npy")
trn_label_DGSSH = np.load("./model_ISIC2018(5)/DGSSH/trn_label.npy")
tst_binary_DGSSH = np.load("./model_ISIC2018(5)/DGSSH/tst_binary.npy")
tst_label_DGSSH = np.load("./model_ISIC2018(5)/DGSSH/tst_label.npy")

trn_binary_DAMH = np.load("./model_ISIC2018(5)/DAMH/trn_binary.npy")
trn_label_DAMH = np.load("./model_ISIC2018(5)/DAMH/trn_label.npy")
tst_binary_DAMH = np.load("./model_ISIC2018(5)/DAMH/tst_binary.npy")
tst_label_DAMH = np.load("./model_ISIC2018(5)/DAMH/tst_label.npy")
img_dir = "../data/ISIC2018(5)/"
# img_dir = "../data/ISIC2018(5)/"
with open("../data/ISIC2018(5)/database_demo.txt", "r") as f:
    trn_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]
# with open("../data/chestX-ray/test_demo.txt", "r") as f:
with open("../data/ISIC2018(5)/test_demo.txt", "r") as f:
    tst_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]


m = 10
n = 6
plt.figure(figsize=(20, 25), dpi=50)  # 20, 25
font_size = 30
correct_font_size = 30
wrong_font_size = 45
# tst_select_index = np.random.permutation(range(tst_binary.shape[0]))[0: m]
# names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL']
query_index = 28


row = 0
query_binary = tst_binary[query_index]
query_label = tst_label[query_index]
# print(query_binary)
# print(query_label)
gnd = (np.dot(query_label, trn_label.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(20, 160, 'FHN', size=font_size)  # FHN
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)


row = 1
query_binary = tst_binary_DPSH[query_index]
query_label = tst_label_DPSH[query_index]
gnd = (np.dot(query_label, trn_label_DPSH.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_DPSH)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(15, 160, 'DPSH', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)


row = 2
query_binary = tst_binary_DBDH[query_index]
query_label = tst_label_DBDH[query_index]
gnd = (np.dot(query_label, trn_label_DBDH.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_DBDH)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(15, 160, 'DBDH', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)


row = 3
query_binary = tst_binary_IDHN[query_index]
query_label = tst_label_IDHN[query_index]
gnd = (np.dot(query_label, trn_label_IDHN.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_IDHN)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(15, 160, 'IDHN', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)

row = 4
query_binary = tst_binary_VTS[query_index]
query_label = tst_label_VTS[query_index]
gnd = (np.dot(query_label, trn_label_VTS.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_VTS)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(20, 160, 'VTS', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)

row = 5
query_binary = tst_binary_SADH[query_index]
query_label = tst_label_SADH[query_index]
gnd = (np.dot(query_label, trn_label_SADH.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_SADH)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(10, 160, 'SADH', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)

row = 6
query_binary = tst_binary_DPN[query_index]
query_label = tst_label_DPN[query_index]
gnd = (np.dot(query_label, trn_label_DPN.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_DPN)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(20, 160, 'DPN', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)


row = 7
query_binary = tst_binary_HybridHash[query_index]
query_label = tst_label_HybridHash[query_index]
gnd = (np.dot(query_label, trn_label_HybridHash.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_HybridHash)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(-20, 160, 'HybridHash', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)


row = 8
query_binary = tst_binary_DGSSH[query_index]
query_label = tst_label_DGSSH[query_index]
gnd = (np.dot(query_label, trn_label_DGSSH.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_DGSSH)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(0, 160, 'DGSSH', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)


row = 9
query_binary = tst_binary_DAMH[query_index]
query_label = tst_label_DAMH[query_index]
gnd = (np.dot(query_label, trn_label_DAMH.transpose()) > 0).astype(np.float32)
hamm = CalcHammingDist(query_binary, trn_binary_DAMH)
ind = np.argsort(hamm)[:n]
t_gnd = gnd[ind]
q_hamm = hamm[ind].astype(int)
q_img_path = tst_img_path[query_index]
return_img_list = np.array(trn_img_path)[ind].tolist()
# print(return_img_list)
plt.subplot(m, n + 1, row * (n + 1) + 1)
img = Image.open(q_img_path).convert('RGB').resize((128, 128))
plt.imshow(img)
plt.axis('off')
# for idx in range(5):
# print(names[idx])
plt.text(5, 160, 'DAMH', size=font_size)
for index, img_path in enumerate(return_img_list):
    # plt.subplot(1, n + 1, index + 2)
    plt.subplot(m, n + 1, row * (n + 1) + index + 2)
    img = Image.open(img_path).convert('RGB').resize((120, 120))
    if t_gnd[index]:
        plt.text(55, 160, '√', size=correct_font_size)
        img = ImageOps.expand(img, 4, fill=(0, 0, 255))
    else:
        plt.text(55, 160, '×', size=wrong_font_size)
        img = ImageOps.expand(img, 4, fill=(255, 0, 0))
    plt.axis('off')
    plt.imshow(img)

plt.subplots_adjust(
    hspace=0.3,
)
plt.savefig("demo_ISIC.png", dpi=200)
plt.show()

