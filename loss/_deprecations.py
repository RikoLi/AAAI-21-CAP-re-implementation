# Naive implementations of camera-aware losses,
# deprecated and will no longer be used.


import numpy as np
import torch

def intra_cam_loss(features, camids, proxy_labels, memory, temp=0.07):
    cam_set = sorted(set(camids))
    camids = np.array(camids)
    intra_loss = 0

    # import ipdb; ipdb.set_trace()
    for cam_name in cam_set:
        sample_num = np.argwhere(cam_name==camids).reshape(-1).size
        indices = np.where(cam_name==camids)[0]
        loss_in_same_cam = 0

        for i in indices:
            # numerator
            sim = torch.dot(features[i], memory.storage[cam_name][proxy_labels[i]])
            sim = torch.exp(sim / temp)
            
            # denominator
            deno = 0
            for j in indices:
                deno_sim = torch.dot(features[i], memory.storage[cam_name][proxy_labels[j]])
                deno += torch.exp(deno_sim / temp)

            # log softmax
            loss_in_same_cam += torch.log(sim / deno)

        intra_loss += loss_in_same_cam / sample_num
    
    return -intra_loss / len(cam_set) # mean loss on all cameras

def inter_cam_loss(k, features, camids, cluster_labels, all_labels, memory, temp=0.07):
    bsize = features.size(0)
    inter_loss = 0

    for i in tqdm.tqdm(range(bsize)):
        # import ipdb; ipdb.set_trace()
        # retrieve all positive samples of i-th feature
        pos_proxies = _retrieve_all_positive_proxies(features, cluster_labels, camids, all_labels, memory)
        neg_proxies = _retrieve_k_nearest_negative_proxies(k, features, cluster_labels, all_labels, memory)

        pos_loss = 0
        pos_cardinality = pos_proxies.size(0)
        neg_cardinality = neg_proxies.size(0)
        for j in tqdm.tqdm(range(pos_cardinality)):
            sim = torch.dot(features[i], pos_proxies[j])
            sim = torch.exp(sim / temp)

            deno = 0
            for p in range(pos_cardinality):
                deno_p_sim = torch.dot(features[i], pos_proxies[p])
                deno_p_sim = torch.exp(deno_p_sim / temp)
                deno += deno_p_sim
            for q in range(neg_cardinality):
                deno_q_sim = torch.dot(features[i], neg_proxies[q])
                deno_q_sim = torch.exp(deno_q_sim / temp)
                deno += deno_q_sim
            
            pos_loss += torch.log(sim / deno)

        inter_loss += pos_loss / pos_cardinality
    
    return -inter_loss / bsize # mean loss on a batch

def _knn(k, x, features):
    '''Find k-nearest neighbors.'''
    diff = torch.norm((features - x), dim=1, keepdim=True)
    k_nearest_indices = diff.view(-1).sort()[1][:k]
    return features[k_nearest_indices]