import torch
import numpy as np

class PBSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, proxy_num=8, samples_per_proxy=4):
        super(PBSampler, self).__init__(dataset)
        self.dataset = dataset
        self.proxy_num = proxy_num
        self.samples_per_proxy = samples_per_proxy

    def __iter__(self):
        batch_size = self.proxy_num * self.samples_per_proxy
        iters = len(self.dataset) // batch_size

        iterator = []
        all_abs_proxy_labels = np.array([sample[-1] for sample in self.dataset.samples])
        all_proxies = np.unique(all_abs_proxy_labels)
        indices = np.where(all_proxies != -1)[0] # remove outliers
        all_proxies = all_proxies[indices]

        for i in range(iters):
            chosen_proxies = np.random.choice(all_proxies, self.proxy_num)
            for p in chosen_proxies:
                indices = np.where(all_abs_proxy_labels == p)[0]
                chosen_ids = np.random.choice(indices, self.samples_per_proxy)
                iterator.extend(chosen_ids.tolist())
        return iter(iterator)

    def __len__(self):
        return len(self.dataset)

class InlierSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        super(InlierSampler, self).__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        all_abs_proxy_labels = np.array([sample[-1] for sample in self.dataset.samples])
        indices = np.where(all_abs_proxy_labels != -1)[0] # remove outliers
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)

class ProxySampler(torch.utils.data.Sampler):
    def __init__(self, dataset, k=4):
        super(ProxySampler, self).__init__(dataset)
        self.dataset = dataset
        self.k = k

    def __iter__(self):
        all_abs_proxy_labels = np.array([sample[-1] for sample in self.dataset.samples])
        indices = np.where(all_abs_proxy_labels != -1)[0] # remove outliers
        inlier_proxies = all_abs_proxy_labels[indices]
        uniq_proxies = np.unique(inlier_proxies)

        iterator = []
        np.random.shuffle(uniq_proxies)
        for proxy in uniq_proxies:
            indices = np.where(all_abs_proxy_labels == proxy)[0]
            indices = np.random.choice(indices, self.k)
            iterator.extend(indices.tolist())
        return iter(iterator)
    
    def __len__(self):
        return len(self.dataset)