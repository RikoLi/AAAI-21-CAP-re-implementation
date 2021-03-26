from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tools.faiss_rerank import compute_jaccard_distance
import torch
import os
import numpy as np

# Rebuild based on SpCL clustering stategy
# Reference: https://github.com/yxgeee/SpCL/blob/master/examples/spcl_train_usl.py
class Clusterer:
    '''
    A DBSCAN clusterer with features.

    Args:
        features: Pytoch tensor from backbone network (L2-normalized).
        eps: float, eps of DBSCAN.
        min_samples: int, mininum neighboring samples of DBSCAN.
        is_cuda: boolean, whether to use GPU acceleration.
        k1: int, k1 in jaccard distance computation.
        k2: int, k2 in jaccard distance computation.
    
    Returns:
        A DBSCAN clusterer.
    '''
    def __init__(self, features, eps, min_samples=4, is_cuda=False, k1=20, k2=6):
        self.features = features.clone() # get a copy
        self.eps = eps
        self.min_samples = min_samples
        self.is_cuda = is_cuda
        self.k1 = k1
        self.k2 = k2
        self.rerank_dist = None
        self.clusterer = None

    def _init_rerank_dist(self):
        search_option = -1 # use faiss-cpu
        if self.is_cuda:
            search_option = 0 # use faiss-gpu
        # NOTE 2021.3.25 Avoid OOM exceptions
        try:
            self.rerank_dist = compute_jaccard_distance(self.features, k1=self.k1, k2=self.k2, search_option=search_option)
        except:
            print('>>> OOM exception, use faiss-cpu mode.')
            self.rerank_dist = compute_jaccard_distance(self.features, k1=self.k1, k2=self.k2, search_option=-1)
        # del self.features

    def _init_dbscan_cluster(self):
        self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed', n_jobs=-1)

    def _compute_centroids(self, pseudo_labels, num_ids):
        centroids = []
        for i in range(num_ids):
            indices_in_same_cluster = np.argwhere(pseudo_labels==i)
            centroid = 0
            for idx in indices_in_same_cluster:
                centroid += self.features[idx].squeeze()
            centroid /= len(indices_in_same_cluster) # average vector
            centroid /= centroid.norm() # L2 normalization
            centroids.append({'cluster_id': i, 'feature': centroid})
        return centroids

    def cluster(self, visualize_path=None, epoch=None):
        '''
        Clustering on given data.

        Args:
            visualize_path: str, path to save clustering visualization results.

        Returns:
            pseudo_labels: Pseudo labels of features.
            num_ids: Number of different clusters.
            centroids: List of centroids of all clusters.
        '''
        print('>>> Start global clustering on total {} samples...'.format(self.features.shape[0]))
        self._init_rerank_dist()
        self._init_dbscan_cluster()
        pseudo_labels = self.clusterer.fit_predict(self.rerank_dist) # generate pseudo labels
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0) # centroid numbers
        centroids = self._compute_centroids(pseudo_labels, num_ids)
        if visualize_path is not None and epoch is not None:
            self._visualize(pseudo_labels, num_ids, visualize_path, epoch)
        print('>>> Found {} clusters.'.format(num_ids))
        return pseudo_labels, num_ids, centroids

    def _visualize(self, pseudo_labels, num_ids, visualize_path, epoch):
        x = self.features.cpu().detach().numpy()
        x = PCA(n_components=50).fit_transform(x) # dim reduction
        x = TSNE(n_components=2, init='pca').fit_transform(x) # project to 2D plane

        # plot noises
        noise_indices = np.argwhere(pseudo_labels==-1)
        plt.clf()
        plt.scatter(x[noise_indices,0], x[noise_indices,1], c='black', marker='.', alpha=0.1)

        # plot inliers
        for cluster_id in range(num_ids):
            indices = np.argwhere(pseudo_labels==cluster_id)
            color = np.random.rand(1,3)
            color = np.repeat(color, len(indices), axis=0)
            plt.scatter(x[indices,0], x[indices,1], c=color, marker='.', alpha=0.5)
        
        plt.savefig(os.path.join(visualize_path, 'cluster_epoch_{}.png'.format(epoch)), dpi=300)
        print('>>> Clustering result is saved to {}.'.format(visualize_path))