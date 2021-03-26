import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd

__all__ = ['ProxyMemoryBank', 'get_abs_proxy_labels']

def get_abs_proxy_labels(camids, proxy_labels, cam_proxy_map):
    '''
    Retrieve absolute proxy labels with camera indices, relative proxy labels and the cam-proxy mappings.
    
    Args:
        camids: list, camera indices.
        proxy_labels: tensor, relative proxy labels.
        cam_proxy_map: dict, record of cam-proxy mappings.

    Returns:
        A tensor of absolute proxy labels.
    '''
    res = []
    for cid, plabel in zip(camids, proxy_labels):
        proxy_cls_label = cam_proxy_map[cid]['cam_index'] + plabel
        res.append(proxy_cls_label)
    res = torch.tensor(res)
    # if torch.cuda.is_available():
    #     res = res.cuda()
    return res

class ProxyMemoryBank(nn.Module):
    '''
    Memory bank of proxy feature centroids.

    ProxyMemoryBank.storage records all proxy centroids. It is a tensor with the shape of (proxy_num, feature_dim).

    Args:
        features_dims: int, dimensions of each feature stored in the bank.
        cam_proxy_map: dict, record of cam-proxy mappings.
        momentum: float, momentum factor in updating of the memory bank.
    '''
    def __init__(self, feature_dims, cam_proxy_map, momentum=0.2):
        super(ProxyMemoryBank, self).__init__()
        self.feature_dims = feature_dims
        self.cam_proxy_map = cam_proxy_map
        self.momentum = momentum
        self.register_buffer('storage', self._init_storage()) # memory bank storage structure, implemented with pytorch register_buffer()

        # NOTE for debugging
        self.filename_record = [None for i in range(self.storage.size(0))]
    
    def update(self, features, abs_proxy_labels):
        '''
        Update the memory bank with given features.

        The update goes like:
        
            saved_feature = momentum * saved_feature + (1 - momentum) * input_feature

        Args:
            features: tensor, containing all features extracted by the backbone model.
            abs_proxy_labels: tensor, absolute proxy labels of the features.
        '''
        features = features.clone().detach()
        abs_proxy_labels = abs_proxy_labels.clone().detach()
        for feat, abs_proxy_label in zip(features, abs_proxy_labels):
            self.storage[abs_proxy_label] = self.momentum * self.storage[abs_proxy_label] + (1 - self.momentum) * feat
            self.storage[abs_proxy_label] /= self.storage[abs_proxy_label].norm()

    def init_storage(self, dataset, features):
        '''
        Initialize proxy memory bank entities with given dataset and features. Both
        two arguments share the same order of samples. After the initialization, the
        ProxyMemoryBank.storage is filled with proxy centroids.

        Args:
            dataset: ProxyDataset, containing all refined samples after clustering.
            features: tensor, containing all features extracted by the backbone model.
        '''
        default_order_map = dataset.default_order_map

        # Step 1: enumerate on each camera
        for camid, item in self.cam_proxy_map.items():
            same_cam_samples = self._get_same_cam_samples(camid, dataset)
            
            # Step 2: get each proxy's centroid and initialize memory
            for i in range(item['proxy_num']):
                centroid = self._cal_proxy_centroid(same_cam_samples, i, features, default_order_map)
                # self.storage[camid][i,:] = centroid
                self.storage[i+dataset.cam_proxy_map[camid]['cam_index'],:] = centroid

                # NOTE record file name
                self.filename_record[i+dataset.cam_proxy_map[camid]['cam_index']] = [item[2] for item in same_cam_samples]

    def _get_same_cam_samples(self, camid, dataset):
        # indices_and_pid = [(idx, sample[4]) for idx, sample in enumerate(dataset.samples) if sample[2] == camid] # sample[4] == proxy_label, sample[2] == cam_id
        # NOTE 加入file name
        indices_and_pid = [(idx, sample[4], sample[1]) for idx, sample in enumerate(dataset.samples) if sample[2] == camid] # sample[4] == proxy_label, sample[2] == cam_id
        return indices_and_pid

    def _cal_proxy_centroid(self, same_cam_samples, proxy_id, features, default_order_map):
        feature_proposals = []
        for idx, proxy_label, fname in same_cam_samples:
            if proxy_label == proxy_id:
                feature_proposals.append(features[default_order_map[idx],:].clone().detach().view(1, -1))
        feature_proposals = torch.cat(feature_proposals, dim=0)
        res = torch.mean(feature_proposals, dim=0, keepdim=True)
        return torch.nn.functional.normalize(res, dim=1)


    def _init_storage(self):
        storage = []
        for camid, item in self.cam_proxy_map.items():
            storage.append(torch.zeros((item['proxy_num'], self.feature_dims)))
        return torch.cat(storage, dim=0)