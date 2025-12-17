import pickle
import os
import argparse
import logging
import torch
import time
from einops import rearrange
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
from sklearn.cluster import MeanShift, estimate_bandwidth
from utils.ViT_Fuzzy import Embedding
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.fuzzy_clustering import ESSC, FuzzyCMeans
import utils.data_processing as dp
import utils.adsh_loss_center as al
import utils.ViT_Fuzzy as vit
import utils.FNN_center as fnn
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore",  "You  are  using  torch.load  with  weights_only=False",  category=FutureWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description="ViTHashNet demo")
parser.add_argument('--bits', default='8', type=str,
                    help='binary code length (default: 12,24,36,48)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 0)')
parser.add_argument('--arch', default='ViT', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=40, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=32, type=int,
                    help='batch size (default: 32)')
parser.add_argument('--num-samples', default=5000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=0.1, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--learning-rate', default=0.0001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')


def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return


def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return


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


def cal_attention1(data):
    n_smpl, n_fea, n_rule = data.shape
    scale_factor = n_fea ** -0.5
    sample_data = torch.from_numpy(data)
    # norm = nn.LayerNorm(n_fea)
    data_list = torch.empty(n_smpl, n_fea, n_rule, dtype=sample_data.dtype)
    for i in range(n_rule):
        q = sample_data[:, :, i]
        attn = (q @ q.transpose(-1, -2)) * scale_factor
        attn = attn.softmax(dim=-1)
        data_per = attn @ q  # torch.Size([n_smpl, n_fea])
        data_per = q + data_per
        norm = nn.LayerNorm(normalized_shape=n_fea)
        data_per = data_per.to(torch.float32)
        data_per = norm(data_per)
        data_per = data_per.to(torch.float64)
        data_list[:, :, i] = data_per
    data_list_np = data_list.detach().numpy()
    return data_list_np


def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S


def calc_loss(V, U, S, code_length, select_index, gamma, labels, is_single=1):
    num_database = V.shape[0]
    # square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    V = Variable(torch.from_numpy(V).type(torch.FloatTensor).cuda())
    V_omega = V[select_index, :]
    u = torch.tensor(U).cuda()
    u = u.tanh()
    hash_center = []
    if is_single == 1:
        category_labels = np.argmax(labels, axis=1)
        n_clusters = len(np.unique(category_labels))
        H_per_list = []
        sort_labels = np.unique(category_labels)
        sort_labels = np.sort(sort_labels)
        for i in sort_labels:
            H_per_list.append(u[category_labels == i])
        C = []
        for H_per in H_per_list:
            fuzzy_cluster1 = FuzzyCMeans(1)
            fuzzy_cluster1.fit(H_per.cpu())
            C_per = fuzzy_cluster1.center_
            C.append(C_per)
        C = np.vstack(C)
        x = labels.argmax(axis=1)
        contains_zero = 0 in x
        if contains_zero is False:
            x -= 1
        has_missing_number = True
        while has_missing_number:
            unique_numbers = torch.unique(x)
            missing_number = torch.arange(torch.min(unique_numbers), torch.max(unique_numbers) + 1).tolist()
            missing_number = next((x for x in missing_number if x not in unique_numbers.tolist()), None)
            if missing_number is not None:
                x[x > missing_number] -= 1
            else:
                has_missing_number = False

        hash_center = C[x]
        hash_center = torch.from_numpy(hash_center).cuda()
    else:
        bandwidth = estimate_bandwidth(V, quantile=0.2, n_samples=5000)
        mean_shift_cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        mean_shift_cluster.fit(V)
        C = mean_shift_cluster.cluster_centers_
        n_clusters = len(np.unique(mean_shift_cluster.labels_))
    criterion = torch.nn.BCELoss()
    square_loss = criterion(0.5 * (u.to(torch.float64) + 1), 0.5 * (hash_center + 1))
    quantization_loss = (torch.tensor(U).cuda()-V_omega) ** 2
    balance_loss = 0.05 * (V_omega.mean(axis=0) ** 2)
    loss = square_loss.sum() + (gamma * quantization_loss.sum() + balance_loss.sum()) / opt.num_samples
    return loss


def encode(model, data_loader, num_data, bit, fea, kkk):
    B = np.zeros([num_data, bit], dtype=np.float64)
    for iter, data in enumerate(data_loader, 0):
        _, _, data_ind = data
        data_input = fea[data_ind.cpu().numpy(), :]  # [b, n_features=6912, n_clusters]
        data_input = torch.from_numpy(data_input)
        data_input = Variable(data_input.cuda())
        output, output_cls = model(data_input, kkk)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B


def adjusting_learning_rate(optimizer, iter):
    update_list = [20, 25, 30]
    # update_list = [15, 25, 35, 40]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
            print('The learning rate of this iter is: ', param_group['lr'])


def adsh_algo(code_length):
    kkk = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # torch.manual_seed(5000)
    torch.manual_seed(5000)
    torch.cuda.manual_seed(5000)
    is_single = 1
    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma
    record['param']['topk'] = [1, 10, 50, 100, 200, 300, 400, 500]
    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels
    output_dir = "D:\\FHN\\demo\\ISIC2018(5)\\float32"
    with open(os.path.join(output_dir, 'database_images.pkl'), 'rb') as f:
        image_databaser = pickle.load(f).cpu()  # [46423, 6912]
    scaler = StandardScaler()
    aaaa = image_databaser
    image_databaser = scaler.fit_transform(image_databaser.detach().numpy())
    image_databaser = torch.tensor(image_databaser)
    # print(image_databaser.shape)
    with open(os.path.join(output_dir, 'test_images.pkl'), 'rb') as f:
        image_testr = pickle.load(f).cpu()
    image_testr = scaler.fit_transform(image_testr.detach().numpy())
    image_testr = torch.tensor(image_testr)
    '''
    model construction
    '''
    # ==================================================================================================================
    n_clusters = 3
    fuzzy = fnn.FNN(n_clusters, cluster_m=2, cluster_eta=0.01, cluster_gamma=0.01, cluster_scale=3)
    fuzzy_database, rule_center, var = fuzzy.fuzzy_layer(image_databaser)
    # print('var', var)
    # print('rule_center', rule_center)
    # vit
    model = vit.ViT(rule_center, var, code_length, n_clusters, cls=5, image_size=384, patch_size=128, channels=3, dim=768, depth=6, heads=8, mlp_dim=1024)
    # print(model)
    model.cuda()
    adsh_loss = al.ADSHLoss(gamma, code_length, num_database, is_single)
    criteria_single = nn.CrossEntropyLoss()
    criteria = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    V = np.zeros((num_database, code_length))
    best_map = 0
    best_lr = 0
    best_topk = [[0 for _ in range(2)] for _ in range(8)]

    model.train()
    for iter in range(max_iter):
        iter_time = time.time()
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        # select_index = list(np.arange(0, 5000))
        # print(select_index)
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size, sampler=_sampler, shuffle=False, num_workers=0)
        '''
        learning deep neural network: feature learning
        '''
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_samples, code_length), dtype=np.float64)
        # rule_center = []
        if iter == 0:
            fuzzy_database = fuzzy_database[:, :-1, :]
            sample_database = fuzzy_database[select_index]
            result_database = fuzzy_database
            result_database[select_index] = cal_attention1(sample_database)

            mean_test = fuzzy.__firing_level__(image_testr, rule_center, var)
            fuzzy_testr = fuzzy.x2xp(image_testr, mean_test)
            fuzzy_testr = fuzzy_testr[:, :-1, :]
            result_test = cal_attention1(fuzzy_testr)

        else:
            mean_database = fuzzy.compute_fire(image_databaser, model.center.cpu(), model.var.cpu())
            # print('mean_database', mean_database)
            fuzzy_database = fuzzy.x2xp(image_databaser, mean_database)
            fuzzy_database = fuzzy_database[:, :-1, :]
            sample_database = fuzzy_database[select_index]
            result_database = fuzzy_database
            result_database[select_index] = cal_attention1(sample_database)

            mean_test = fuzzy.compute_fire(image_testr, model.center.cpu(), model.var.cpu())
            # print('model.center', model.center)
            # print('mean_test', mean_test)
            fuzzy_testr = fuzzy.x2xp(image_testr, mean_test)
            fuzzy_testr = fuzzy_testr[:, :-1, :]
            result_test = cal_attention1(fuzzy_testr)

        loss_train = 0
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                print('image_databaser', image_databaser[batch_ind.cpu().numpy(), :])
                kkk = 1
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                fuzzy_input = result_database[batch_ind.cpu().numpy(), :]
                fuzzy_input = torch.from_numpy(fuzzy_input)
                fuzzy_input = Variable(fuzzy_input.cuda())
                output, output_cls = model(fuzzy_input, kkk)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                U[u_ind, :] = output.cpu().data.numpy()
                model.zero_grad()
                loss = adsh_loss(output, V, S, V[batch_ind.cpu().numpy(), :], database_labels, train_label, batch_ind)
                cls_loss = 0.
                for i in range(train_label.shape[0]):
                    if train_label[i].sum() > 1:
                        cls_loss += criteria(output_cls[i].float().reshape(output_cls[i].shape[0], 1), train_label[i].float().reshape(train_label[i].shape[0], 1).cuda())
                    else:
                        cls_loss += criteria_single(output_cls[i].float().reshape(output_cls[i].shape[0], 1), train_label[i].float().reshape(train_label[i].shape[0], 1).cuda())
                loss = loss + 0.1 * cls_loss
                loss_train += loss.item()
                loss.backward()
                optimizer.step()
        adjusting_learning_rate(optimizer, iter)
        lr_adj = optimizer.param_groups[0]['lr']

        '''
        learning binary codes: discrete coding
        '''
        barU = np.zeros((num_database, code_length))
        barU[select_index, :] = U
        Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU

        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        iter_time = time.time() - iter_time
        print('iter_time:', iter_time)
        loss_train = loss_train / (epochs * len(trainloader))

        logger.info('[Iteration: %3d/%3d][Train Loss: %.3f]', iter, max_iter, loss_train)
        record['train loss'].append(loss_train)
        record['iter time'].append(iter_time)

        if (iter+1) % 2 == 0:
            print('This iter is:', iter)
            '''
            training procedure finishes, evaluation
            '''
            model.eval()

            torch.save(model.state_dict(), os.path.join('./', 'demo/model_ISIC2018(5).pt'))

            testloader = DataLoader(dset_test, batch_size=32, shuffle=False, num_workers=0)
            qB = encode(model, testloader, num_test, code_length, result_test, kkk)
            rB = V
            map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
            np.save(os.path.join('./', 'demo/test_binary_ISIC2018(5).npy'), qB)
            np.save(os.path.join('./', 'demo/database_binary_ISIC2018(5).npy'), rB)
            logger.info('[Evaluation: mAP: %.3f]', map)
            record['map'] = map
            t = 0
            for topk in record['param']['topk']:
                topkmap = calc_hr.calc_topMap(qB, rB, test_labels.numpy(), database_labels.numpy(), topk)
                logger.info('[top-%d mAP: %.3f]', topk, topkmap)
                record['topkmap'] = topkmap
                if topkmap > best_topk[t][0]:
                    best_topk[t][0] = topkmap
                    best_topk[t][1] = lr_adj
                t = t + 1
            if map > best_map:
                best_map = map
                best_lr = lr_adj
            record['rB'] = rB
            record['qB'] = qB

            filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')
            _save_record(record, filename)
    print('best_map', best_map)
    print('best_lr', best_lr)
    print('best_topk', best_topk)


if __name__ == "__main__":
    opt = parser.parse_args()
    logdir = '-'.join(['log/ViTHashNet-ISIC2018(5)_fuzzy_attention_center', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    print('ViTHashNet:')
    for count in range(1):
        print('=======================================================================================================')
        print('count:', count)
        for bit in bits:
            adsh_algo(bit)
