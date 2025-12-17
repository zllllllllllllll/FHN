import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.fuzzy_clustering import ESSC, FuzzyCMeans


class ADSHLoss(nn.Module):
    def __init__(self, gamma, code_length, num_train, is_single):
        super(ADSHLoss, self).__init__()
        self.gamma = gamma
        self.code_length = code_length
        self.num_train = num_train
        self.is_single = is_single
        self.multi_label_random_center = torch.randint(2, (code_length,)).double()

    def forward(self, u, V, S, V_omega, labels, train_label, batch_ind):
        batch_size = u.size(0)
        u = u.tanh()
        u = u.cuda()
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())  # V_omega cuda torch float32
        if self.is_single == 1:
            category_labels = np.argmax(train_label, axis=1)
            V_per_list = []
            sort_labels = np.unique(category_labels)
            for i in sort_labels:
                V_per_list.append(u[category_labels == i])
            C = []
            for V_per in V_per_list:
                fuzzy_cluster1 = FuzzyCMeans(1)
                fuzzy_cluster1.fit(V_per.cpu())
                C_per = fuzzy_cluster1.center_
                C.append(C_per)
            C = np.vstack(C)
            del_label = train_label.numpy()
            zero_columns = np.all(del_label == 0, axis=0)
            del_label = train_label[:, ~zero_columns]
            x = del_label.argmax(axis=1)
            hash_center = C[x]
            hash_center = torch.from_numpy(hash_center).cuda()
        else:
            num_classes = train_label.shape[1]
            C = []
            for i in range(num_classes):
                indices = np.where(train_label[:, i] == 1)[0]
                if indices.size > 0:
                    H_per = u[indices]
                    fuzzy_cluster1 = FuzzyCMeans(1)
                    fuzzy_cluster1.fit(H_per.cpu())
                    C_per = fuzzy_cluster1.center_
                    C.append(C_per)
            C = np.vstack(C)
            del_label = train_label.numpy()
            zero_columns = np.all(del_label == 0, axis=0)
            del_label = train_label[:, ~zero_columns]
            center_sum = del_label @ np.array(C)
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = F.softsign(center_sum).cuda()

        criterion = torch.nn.BCELoss()
        square_loss = criterion(0.5 * (u.to(torch.float64) + 1), 0.5 * (hash_center + 1))
        quantization_loss = self.gamma * (V_omega - u) ** 2
        quantization_loss = quantization_loss.sum() / batch_size
        balance_loss = 0.003 * (V_omega.mean(axis=0) ** 2)
        balance_loss = balance_loss.sum() / batch_size
        loss = square_loss + quantization_loss + balance_loss
        return loss