import numpy as np
from utils.fuzzy_clustering import ESSC, FuzzyCMeans
import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from numpy import linalg as la
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, \
    adjusted_rand_score
# import skfuzzy as fuzz
# device = 'cpu'  # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class FNN(nn.Module):
    def __init__(self, n_clusters, cluster_m=2, cluster_eta=0.01, cluster_gamma=0.01, cluster_scale=1):

        """
        :param data: data
        :param n_clusters: number of clusters
        :param cluster_m: fuzzy index $m$ for ESSC
        :param cluster_eta: parameter $\eta$ for ESSC
        :param cluster_gamma: parameter $\gamma$ for ESSC
        :param cluster_scale: scale parameter for ESSC
        """

        super(FNN, self).__init__()

        self.n_clusters = n_clusters
        self.cluster_m = cluster_m
        self.cluster_eta = cluster_eta
        self.cluster_gamma = cluster_gamma
        self.cluster_scale = cluster_scale
        self.alpha = 1
        self.cluster_centers = None
        self.fc1 = nn.ModuleList()
        self.params_ = None
        self.is_initialized_ = False

    def get_params(self, deep=True):
        return {
            'n_clusters': self.n_clusters,
            'cluster_m': self.cluster_m,
            'cluster_eta': self.cluster_eta,
            'cluster_gamma': self.cluster_gamma,
            'cluster_scale': self.cluster_scale,
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fuzzy_layer(self, X):
        self.center, self.var = self.__cluster__(X, self.n_clusters, self.cluster_eta, self.cluster_gamma,
                                                 self.cluster_scale)
        mem = self.__firing_level__(X, self.center, self.var)
        Xp = self.x2xp(X, mem)

        return Xp, self.center, self.var


    @staticmethod
    def __cluster__(data, n_cluster, eta, gamma, scale):
        """
        Comute data centers and membership of each point by ESSC, and compute the variance of each feature
        :param data: n_Samples * n_Features
        :param n_cluster: number of center
        :return: centers: data center, delta: variance of each feature
        """
        # fuzzy_cluster = ESSC(n_cluster, eta=eta, gamma=gamma, tol_iter=100, scale=scale).fit(data)
        # centers = fuzzy_cluster.center_
        # delta = fuzzy_cluster.variance_

        kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(data.detach().numpy())
        centers = kmeans.cluster_centers_
        # Calculate average distance to set widths
        distances = pairwise_distances(centers, centers)
        np.fill_diagonal(distances, np.inf)
        valid_distances = distances[~np.isinf(distances)]
        if valid_distances.size == 0:
            raise ValueError("No valid distances found between cluster centers.")
        widths = np.mean(valid_distances) * np.ones(n_cluster)
        centers = centers
        delta = widths
        return centers, delta


    @staticmethod
    def __firing_level__(data, centers, delta):
        """
        Compute firing strength using Gaussian model
        :param data: n_Samples * n_Features
        :param centers: data center，n_Clusters * n_Features
        :param delta: variance of each feature， n_Clusters * n_Features
        :return: firing strength
        """
        data1 = data.detach().numpy()
        d = -(np.expand_dims(data1, axis=2) - np.expand_dims(centers.T, axis=0)) ** 2 / (2 * delta.T)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)
        return d / np.sum(d, axis=1, keepdims=True)

    @staticmethod
    def compute_fire(data, centers, delta):
        """
        Compute firing strength using Gaussian model
        :param data: n_Samples * n_Features
        :param centers: data center，n_Clusters * n_Features
        :param delta: variance of each feature， n_Clusters * n_Features
        :return: firing strength
        """
        # print('data.device', data.device)
        # print('centers', centers.device # numpy.ndarray
        data1 = data.detach().numpy()
        centers1 = centers.detach().numpy()
        delta1 = delta.detach().numpy()
        d = -(np.expand_dims(data1, axis=2) - np.expand_dims(centers1.T, axis=0)) ** 2 / (2 * delta1.T)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)
        return d / np.sum(d, axis=1, keepdims=True)

    @staticmethod
    def x2xp(X, mem, order=1):
        """
        Converting raw input feature X to TSK consequent input
        :param X: raw input, [n_sample, n_features]
        :param mem: firing level of each rule, [n_sample, n_clusters]
        :param order:order of TSK, 0 or 1
        :return:
        """
        if order == 0:
            return mem
        else:
            N = X.shape[0]
            X = X.detach().numpy()
            mem = np.expand_dims(mem, axis=1)  # [n_sample,  1,  n_clusters]
            X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
            X = np.repeat(X, repeats=mem.shape[1], axis=2)  # [n_sample,  n_features+1,  n_clusters]
            xp = X * mem  # [n_sample,  n_features+1,  n_clusters]
            return xp