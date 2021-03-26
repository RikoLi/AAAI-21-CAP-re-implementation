import numpy as np
import torch, random
from torch.utils.data import Sampler

class PBSampler(Sampler):
    '''
    Proxy-balanced sampler.

    Enable proxy-balanced sampling to alleviate training bias caused by the unbalanced distribution of proxy classes.

    Args:
        dataset: A ProxyDataset object.
        all_labels: dict, labels of all training samples, the output of get_all_sample_labels().
        proxy_num: int, number of proxies in each batch, default 8.
        samples_per_proxy: int, numbers of samples picked from each proxy, default 4.
    '''

    def __init__(self, dataset, all_labels, proxy_num=8, samples_per_proxy=4):
        super(PBSampler, self).__init__(dataset)
        self.dataset = dataset
        self.all_labels = all_labels
        self.proxy_num = proxy_num
        self.samples_per_proxy = samples_per_proxy

    def __iter__(self):
        batch_size = self.proxy_num * self.samples_per_proxy
        iterations = len(self.dataset) // batch_size if len(self.dataset) % batch_size == 0 else len(self.dataset) // batch_size + 1

        iter_list = []
        proxy_labels = self.all_labels['abs_proxy'].view(-1)
        for i in range(iterations):
            proxy_card = len(set(proxy_labels.tolist()))
            chosen_labels = random.choices(range(proxy_card), k=self.proxy_num)
            for l in chosen_labels:
                indices = torch.where(proxy_labels==l)[0]
                ids = random.choices(range(len(indices)), k=self.samples_per_proxy) # 有重复
                iter_list.extend(indices[ids].tolist())
        return iter(iter_list)

    def __len__(self):
        return len(self.dataset)

class ClusterSampler(Sampler):
    def __init__(self, dataset, cluster_num=8, samples_per_cluster=4):
        super(ClusterSampler, self).__init__(dataset)
        self.dataset = dataset
        self.cluster_num = cluster_num
        self.samples_per_cluster = samples_per_cluster

    def __iter__(self):
        # Step 1: choose cluster_num clusters
        batch_size = self.cluster_num * self.samples_per_cluster
        iterations = len(self.dataset) // batch_size if len(self.dataset) % batch_size == 0 else len(self.dataset) // batch_size + 1

        # Step 2: choose several samples from selected clusters
        iter_list = []
        labels = self.dataset.good_labels.reshape(-1)
        for i in range(iterations):
            cluster_cls_num = len(set(labels.tolist()))
            chosen_labels = random.sample(range(cluster_cls_num), min(self.cluster_num, cluster_cls_num)) # better implementation
            for l in chosen_labels:
                indices = torch.where(torch.tensor(labels)==l)[0]
                ids = random.sample(range(len(indices)), min(self.samples_per_cluster, len(indices))) # valid sampling
                iter_list.extend(indices[ids].tolist())
        return iter(iter_list)

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    pass