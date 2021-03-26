import sys
sys.path.append('/home/ljc/works/fast-reid')

import torch
import numpy as np
from tools.dataset_wrapper import ProxyDataset

__all__ = ['split_for_proxy']

def _append_default_order(data, default_order):
    '''Keep default order of the original data.'''
    res = []
    for item, order in zip(data, default_order):
        tmp = list(item)
        tmp.append(order)
        res.append(tmp)

    return res


def split_for_proxy(refined_dataset):
    '''
    Split the dataset into several camera-aware proxies.

    Args:
        refined_dataset: A pytorch Dataset object after dataset refining.

    Returns:
        A pytorch Dataset object, in which each sample contains (img, cluster_label, proxy_label, cam_id) information.
    '''
    train_data = _append_default_order(refined_dataset.train, refined_dataset.default_order)
    cluster_labels = refined_dataset.good_labels

    cam_set = _get_cams(train_data)
    new_samples, proxy_nums = _append_proxy_labels(train_data, cam_set, cluster_labels)
    # print(new_samples, all_proxy_num) # debug
    print('>>> Found {} proxies.'.format(sum([p['proxy_num'] for p in proxy_nums])))
    
    return ProxyDataset(img_shape=refined_dataset.img_shape, samples=new_samples, proxy_nums=proxy_nums, all_train_samples=refined_dataset.all_train_samples)
    
def _get_cams(data):
    '''
    Return a sorted set of camera names.

    Args:
        data: list, a list of all training data, each sample is (frame, vid, camid).

    Returns:
        A sorted set of camera names.
    '''
    s = set()
    for _, _, camid, _ in data:
        s.add(camid)
    return sorted(s, key=lambda x: int(x.split('_')[-1])) # make sure cam set is ordered, easy to debug

def _append_proxy_labels(data, cam_set, cluster_labels):
    '''
    Return a list of dataset samples with pseudo proxy labels and the number of proxies.

    Args:
        data: list, a list of all training data, each sample is (frame, vid, camid).
        cam_set: set, a sorted set of all camera names.
        cluster_labels: list, pseudo cluster labels of corresponding samples.

    Returns:
        Pseudo proxy labels and the number of proxies.
    '''
    results = []
    proxy_nums = []
    cam_slot_index = 0 # global camera index
    # all_proxy_num = 0
    for camid in cam_set:
        same_cam_samples, proxy_num = _get_same_cam_samples(data, camid, cluster_labels) # NOTE: get proxy num in each camera
        # print(proxy_num) # debug
        same_cam_samples = [tuple(item) for item in same_cam_samples]
        results.extend(same_cam_samples)
        proxy_nums.append({'camid': camid, 'proxy_num': proxy_num, 'cam_index': cam_slot_index})
        cam_slot_index += proxy_num
        # all_proxy_num += proxy_num
    return results, proxy_nums

def _get_same_cam_samples(data, camid, cluster_labels):
    '''
    Find proxies with samples of each camera.

    Args:
        data: list, a list of all training data, each sample is (frame, vid, camid).
        camid: str, camera id (name) of current camera.
        cluster_labels: list, pseudo cluster labels of corresponding samples.

    Returns:
        Samples with corresponding pseudo proxy labels of each camera and proxies amount in this camera.
    '''
    results = []
    for data_tuple, cls_label in zip(data, cluster_labels):
        fname, _, cid, default_order = data_tuple # unpacking
        if cid == camid:
            results.append([default_order, fname, cid, cls_label])
    results = sorted(results, key=lambda item: item[-1]) # sort with cluster label
    results, proxy_num = _add_proxies(results)
    return results, proxy_num

def _add_proxies(sorted_samples):
    '''
    Core implementation of _get_same_cam_samples().
    '''
    label = 0
    proxy_num = 0
    for i in range(len(sorted_samples)):
        if i == 0:
            sorted_samples[i].append(0)
            proxy_num += 1
        elif sorted_samples[i][3] != sorted_samples[i-1][3]: # reach proxy boundary, add new label, sorted_samples[i][3] == cluster label
            label += 1
            sorted_samples[i].append(label)
            proxy_num += 1
        else:
            sorted_samples[i].append(label)
    return sorted_samples, proxy_num
        



if __name__ == "__main__":
    from fastreid.data.datasets import VeRi
    from dataset_wrapper import RefinedDataset


    veri = VeRi(root='/home/ljc/datasets')
    veri.train = veri.train[:20] # test with 10 samples
    veri = RefinedDataset((256,256), old_dataset=veri, good_labels=[0,0,0,0,0,1,1,1,2,2,2,2,2,1,1,1,2,2,0,0])

    veri = split_for_proxy(veri)
    print(sorted(veri.samples, key=lambda item: int(item[1].split('_')[-1])))
    print(veri.proxy_num)