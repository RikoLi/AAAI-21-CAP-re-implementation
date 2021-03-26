import torch

__all__ = ['retrieve_all_positive_proxies', 'retrieve_k_nearest_negative_proxies']

def retrieve_all_positive_proxies(input_features, cluster_labels, all_labels, abs_proxy_labels, memory):
    '''
    Retrieve all positive proxies in a batch.

    Args:
        input_features: tensor, a batch of extracted features.
        cluster_labels: tensor, pseudo cluster label of each feature.
        abs_proxy_labels: tensor, absolute proxy centroid indices in the memory bank.
        memory: proxy-level memory bank.

    Returns:
        A tensor of all positive proxies of each sample in the batch.
    '''
    bsize = input_features.size(0)
    res = []
    res_labels = []
    for i in range(bsize):
        x = input_features[i] # i-th feature
        cls_label = cluster_labels[i] # i-th cluster pseudo label
        indices, _ = torch.where(all_labels['cluster']==cls_label) # positive indices
        positive_proxy_indices = sorted(list(set(all_labels['abs_proxy'][indices].numpy().tolist()))) # positive proxy centroid absolute indices
        res.append(memory.storage[torch.tensor(positive_proxy_indices),:]) # positive centroids
        res_labels.append(torch.tensor(list(set(all_labels['abs_proxy'][indices]))))
    return torch.cat(res, dim=0), torch.cat(res_labels)

def retrieve_k_nearest_negative_proxies(k, input_features, cluster_labels, all_labels, abs_proxy_labels, memory):
    '''
    Retrieve k nearest negative proxies as hard-mining samples in inter-camera loss.

    Args:
        k: int, k nearest samples.
        input_features: tensor, a batch of extracted features.
        cluster_labels: tensor, pseudo cluster label of each feature.
        abs_proxy_labels: tensor, absolute proxy centroid indices in the memory bank.
        memory: proxy-level memory bank.

    Returns:
        A tensor of negative proxies with shape of (k*bsize, feature_dim).
    '''
    bsize = input_features.size(0)
    res = []
    res_labels = []
    for i in range(bsize):
        x = input_features[i] # i-th feature
        cls_label = cluster_labels[i] # i-th cluster pseudo label
        indices, _ = torch.where(all_labels['cluster']!=cls_label)
        negative_proxy_indices = sorted(list(set(all_labels['abs_proxy'][indices].numpy().tolist())))
        neg_proxy_centroids = memory.storage[torch.tensor(negative_proxy_indices),:]
        res.append(_knn(k, x, neg_proxy_centroids))
        res_labels.append(torch.tensor(list(set(all_labels['abs_proxy'][indices]))))
    return torch.cat(res, dim=0), torch.cat(res_labels)

def _knn(k, x, features):
    '''Find k-nearest neighbors.'''
    diff = torch.norm((features - x), dim=1, keepdim=True)
    k_nearest_indices = diff.view(-1).sort()[1][:k]
    return features[k_nearest_indices]


if __name__ == "__main__":
    # test
    from sklearn.neighbors import NearestNeighbors
    x = torch.rand((1,10))
    feat = torch.rand((6,10))
    dist = _knn(3, x, feat)
    # dist = torch.norm((feat-x), dim=1, keepdim=True)
    print(dist)

    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(feat.cpu().numpy())
    _, indices = neigh.kneighbors(x.cpu().numpy())
    print(feat[torch.tensor(indices).view(-1)])