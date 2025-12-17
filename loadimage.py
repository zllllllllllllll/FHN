import pickle
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import warnings
from torch import nn
from utils.ViT_Fuzzy import Embedding
from torch.utils.data import DataLoader
from  torchvision  import  models
import utils.data_processing as dp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore",  category=FutureWarning,  module="torch.storage")


# def _dataset():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     transformations = transforms.Compose([
#         transforms.Resize(384),  # chest
#         transforms.CenterCrop(384),
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     dset_database = dp.DatasetProcessingChest(
#         'data/chestX-ray', 'database_img.txt', 'database_label.txt', transformations
#     )
#     dset_test = dp.DatasetProcessingChest(
#         'data/chestX-ray', 'test_img.txt', 'test_label.txt', transformations
#     )
#     num_database, num_test = len(dset_database), len(dset_test)
#
#     def load_label(filename, DATA_DIR):
#         label_filepath = os.path.join(DATA_DIR, filename)
#         label = np.loadtxt(label_filepath, dtype=np.int64)
#         return torch.from_numpy(label)
#
#     databaselabels = load_label('database_label.txt', 'data/chestX-ray')
#     np.save(os.path.join('./', 'demo/database_label_chestX-ray.npy'), databaselabels)
#     testlabels = load_label('test_label.txt', 'data/chestX-ray')
#     np.save(os.path.join('./', 'demo/test_label_chestX-ray.npy'), testlabels)
#     dsets = (dset_database, dset_test)
#     nums = (num_database, num_test)
#     labels = (databaselabels, testlabels)
#     return nums, dsets, labels
#
#
# nums, dsets, labels = _dataset()
# num_database, num_test = nums
# dset_database, dset_test = dsets
#
# # image_database = []
# # embedding = Embedding(num_patches=9, min_dim=48, dim=768)  # 48/28/63
# # 数据库
# databaseloader = DataLoader(dset_database, batch_size=64, shuffle=False, num_workers=0)
# output_dir = "D:\\zll\\VitHashNet\\demo"
# image_databasew = torch.empty(0, 6912,  dtype=torch.float64)
# num_features = 3 * 384 * 384
# linear_layer = nn.Linear(num_features, 6912)
# for iteration,  (train_input, _, _) in enumerate(databaseloader):
#     print(iteration)
#     # print(train_input.device)
#     # image = embedding(train_input)
#     # num_features = train_input.size(1) * train_input.size(2) * train_input.size(3)  # [64,3,384,384]
#     train_input = train_input.view(train_input.size(0), -1)  # [64,3*384*384]
#     train_input = linear_layer(train_input)  # [64,6912]
#     # print('image', train_input.shape)
#     image_databasew = torch.cat((image_databasew, train_input), dim=0)
# print(image_databasew.shape)
# with open(os.path.join(output_dir, 'database_images.pkl'), 'wb') as f:
#     pickle.dump(image_databasew, f)
# print('database存OK')
# with open(os.path.join(output_dir, 'database_images.pkl'), 'rb') as f:
#     image_databaser = pickle.load(f)
# print(image_databaser.shape)
# print('database读OK')
#
# # 测试集
# testloader = DataLoader(dset_test, batch_size=64, shuffle=False, num_workers=0)
# image_testw = torch.empty(0, 6912,  dtype=torch.float64)
# for iteration,  (test_input, _, _) in enumerate(testloader):
#     print(iteration)
#     # print(train_input.device)
#     # image = embedding(train_input)
#     # num_features = train_input.size(1) * train_input.size(2) * train_input.size(3)  # [64,3,384,384]
#     test_input = test_input.view(test_input.size(0), -1)
#     test_input = linear_layer(test_input)  # [64,768]
#     # print('image', train_input.shape)
#     image_testw = torch.cat((image_testw, test_input), dim=0)
# print(image_testw.shape)
# with open(os.path.join(output_dir, 'test_images.pkl'), 'wb') as f:
#     pickle.dump(image_testw, f)
# print('test存OK')
# with open(os.path.join(output_dir, 'test_images.pkl'), 'rb') as f:
#     image_testr = pickle.load(f)
# print(image_testr.shape)
# print('test读OK')




def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(384),  # chest
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessingChest(
        'data/ISIC2018(5)', 'database_img.txt', 'database_label.txt', transformations
    )
    dset_test = dp.DatasetProcessingChest(
        'data/ISIC2018(5)', 'test_img.txt', 'test_label.txt', transformations
    )
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        label_filepath = os.path.join(DATA_DIR, filename)
        label = np.loadtxt(label_filepath, dtype=np.int64)
        return torch.from_numpy(label)

    databaselabels = load_label('database_label.txt', 'data/ISIC2018(5)')
    np.save(os.path.join('./', 'demo/database_label_ISIC2018(5).npy'), databaselabels)
    testlabels = load_label('test_label.txt', 'data/ISIC2018(5)')
    np.save(os.path.join('./', 'demo/test_label_ISIC2018(5).npy'), testlabels)
    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels


nums, dsets, labels = _dataset()
num_database, num_test = nums
dset_database, dset_test = dsets

# image_database = []
embedding = Embedding(num_patches=9, min_dim=48, dim=768)  # 48/28/63

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,  6912)
model.eval()

# database
databaseloader = DataLoader(dset_database, batch_size=1, shuffle=False, num_workers=0)
output_dir = "D:\\zll\\VitHashNet\\demo\\ISIC2018(5)\\embedding"

image_databasew = torch.empty(0, 6912,  dtype=torch.float32)
num_features = 3 * 384 * 384
linear_layer = nn.Linear(num_features, 6912)
for iteration,  (train_input, _, _) in enumerate(databaseloader):
    print(iteration)
    # print(train_input.device)
    # train_input = embedding(train_input)  # [64,9,768]

    train_input = model(train_input)

    # num_features = train_input.size(1) * train_input.size(2) * train_input.size(3)  # [64,3,384,384]
    # train_input = train_input.view(train_input.size(0), -1)  # [64,9*768]
    # train_input = linear_layer(train_input)  # [64,6912]
    # print('image', train_input.shape)
    image_databasew = torch.cat((image_databasew, train_input), dim=0)

print(image_databasew.shape)
with open(os.path.join(output_dir, 'database_images.pkl'), 'wb') as f:
    pickle.dump(image_databasew, f)
print('database存OK')
with open(os.path.join(output_dir, 'database_images.pkl'), 'rb') as f:
    image_databaser = pickle.load(f)
print(image_databaser.shape)
print('database读OK')

# test
testloader = DataLoader(dset_test, batch_size=64, shuffle=False, num_workers=0)
image_testw = torch.empty(0, 6912,  dtype=torch.float32)
for iteration,  (test_input, _, _) in enumerate(testloader):
    print(iteration)
    # print(train_input.device)
    # test_input = embedding(test_input)  # [64,9,768]

    test_input = model(test_input)

    # num_features = train_input.size(1) * train_input.size(2) * train_input.size(3)  # [64,3,384,384]
    # test_input = test_input.view(test_input.size(0), -1)  # [64,9*768]
    # test_input = linear_layer(test_input)  # [64,768]
    # print('image', train_input.shape)
    image_testw = torch.cat((image_testw, test_input), dim=0)
print(image_testw.shape)
with open(os.path.join(output_dir, 'test_images.pkl'), 'wb') as f:
    pickle.dump(image_testw, f)
print('test存OK')
with open(os.path.join(output_dir, 'test_images.pkl'), 'rb') as f:
    image_testr = pickle.load(f)
print(image_testr.shape)
print('test读OK')
