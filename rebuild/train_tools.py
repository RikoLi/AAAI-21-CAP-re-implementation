from rebuild.dataloaders import GlobalDataset
import tqdm
import torch
import numpy as np

def add_new_labels(global_dataset, pseudo_cluster_labels, cam_labels):
    cam_memory_map = {} # record storage index of each proxy centroid
    accumulated_storage_index = 0
    all_proxy_num = 0

    all_cams = np.unique(cam_labels)
    for cam in all_cams:
        # get pseudo cluster labels under each camera
        indices = np.where(cam == cam_labels)[0]
        same_cam_pclabels = pseudo_cluster_labels[indices]

        # remove outliers and get proxy number
        sorted_same_cam_pclabels = np.sort(same_cam_pclabels)
        normal_indices = np.where(sorted_same_cam_pclabels != -1)[0]
        normal_pclabels = sorted_same_cam_pclabels[normal_indices]
        current_proxy_num = np.unique(normal_pclabels).size
        all_proxy_num += current_proxy_num # count proxy number

        
        # mapping from pseudo cluster label to proxy label
        cluster_proxy_map = {clbl: plbl for plbl, clbl in enumerate(np.unique(normal_pclabels))}

        # debug
        # print('cam {}, {} proxies'.format(cam, current_proxy_num))
        
        # update storage index
        cam_memory_map[cam] = accumulated_storage_index
        accumulated_storage_index += current_proxy_num

        # assign proxy labels for each sample, -1 means outlier
        for idx in indices:
            fname, vid, camid = global_dataset.samples[idx]
            global_dataset.samples[idx] = (fname, vid, camid, pseudo_cluster_labels[idx],
                cluster_proxy_map[pseudo_cluster_labels[idx]] if pseudo_cluster_labels[idx] != -1 else -1)

    print('>>> Found {} proxies.'.format(all_proxy_num))
    
    # assign absolute proxy label for each normal sample, -1 for outliers
    for i in range(len(global_dataset.samples)):
        fname, vid, camid, clabel, plabel = global_dataset.samples[i]
        if clabel == -1:
            new_sample = (fname, vid, camid, clabel, plabel, -1)
        else:
            # absolute proxy label = start_index + proxy_label(offset)
            cid = int(camid.split('_')[-1])
            new_sample = (fname, vid, camid, clabel, plabel, cam_memory_map[cid]+plabel)
        global_dataset.samples[i] = new_sample

    return cam_memory_map


def extract_global_features(img_shape, batch_size, workers, model, dataset, is_cuda=False):
    if is_cuda:
        model = model.cuda()

    # build test dataloader
    dataset = GlobalDataset(img_shape, dataset)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    
    # Containers of features and labels
    features = []
    v_labels = []
    cam_labels = []

    # extract features
    model.eval()
    with torch.no_grad():
        for _, (imgs, _, vids, camids) in enumerate(tqdm.tqdm(test_dataloader, desc='extracting proc')):
            if is_cuda:
                imgs = imgs.cuda()

            batch_feats = model(imgs).detach().cpu()
            
            features.append(batch_feats)
            v_labels.append(vids)
            cam_labels.append(camids)
    
    model.train()

    features = torch.cat(features, dim=0)
    v_labels = torch.cat(v_labels, dim=0)
    cam_labels = torch.cat(cam_labels, dim=0)
    return dataset, features, v_labels, cam_labels