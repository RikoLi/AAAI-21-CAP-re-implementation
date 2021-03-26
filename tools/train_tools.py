import torch
import copy
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from tools.dataset_wrapper import CustomDataset, RefinedDataset

def build_dataloader(img_shape, dataset, batch_size, workers, mode='train'):
    '''
    Build a dataloader with fast re-id style dataset. When enumerating on it, it returns\
    (imgs, fnames, vids, camids) as a tuple.

    Args:
        img_shape: tuple, (height, width) of each input image. Depends on the model.
        dataset: Fast re-id style dataset.
        batch_size: int, batch size of each iteration.
        workers: int, workers of pytorch DataLoader.
        mode: str, 'train' | 'test', decide which dataset to use.

    Returns:
        A dataloader.
    '''

    if mode == 'train':
        custom_dataset = CustomDataset(img_shape, dataset, mode='train')
    elif mode == 'test':
        custom_dataset = CustomDataset(img_shape, dataset, mode='test')
    else:
        raise ValueError('Wrong argument value of mode!')
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

def extract_global_features(img_shape, batch_size, workers, model, dataset, mode='train', is_cuda=False):
    '''
    Extract global features from dataset.

    Args:
        img_shape: tuple, (height, width) of each input image. Depends on the model.
        batch_size: int, batch size of each iteration.
        workers: int, workers of pytorch DataLoader.
        model: Pytorch model.
        dataset: Fast re-id style dataset.
        mode: str, 'train' or 'test'.
        is_cuda: boolean, whether to use GPU.
    
    Returns:
        features: OrderedDict, global features.
        v_labels: OrderedDict, vehicle id labels.
        cam_labels: OrderedDict, camera id labels.
    '''
    if is_cuda:
        model = model.cuda()

    data_loader = build_dataloader(img_shape, dataset, batch_size=batch_size, workers=workers, mode=mode)
    
    # Containers of features and labels
    features = OrderedDict()
    v_labels = OrderedDict()
    cam_labels = OrderedDict()

    model = model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, vids, camids) in enumerate(data_loader):
            if is_cuda:
                imgs = imgs.cuda()
            batch_feats = model(imgs).data.cpu() # extract batch of features
            for fname, batch_feat, vid, camid in zip(fnames, batch_feats, vids, camids):
                features[fname] = batch_feat
                v_labels[fname] = vid
                cam_labels[fname] = camid
    model = model.train()
    return features, v_labels, cam_labels

def merge_features_from_dict(features_dict):
    '''
    Merge features from dict to tensor.

    Args:
        features_dict: OrderedDict, features dict.

    Returns:
        Pytorch tensor of all features in the dict.
    '''
    tensor = torch.cat([v.unsqueeze(0) for _, v in features_dict.items()], dim=0)
    return tensor

def refine_dataset(img_shape, dataset, pseudo_labels):
    '''
    Refine original dataset with pseudo labels from clustering. Remove outliers.

    Args:
        img_shape: tuple, expected input image shape.
        dataset: Fast re-id style dataset.
        pseudo_labels: ndarray, output of clustering.

    Returns:
        Refined fast re-id dataset with outliers (-1 labeled) removed. When enumerating on its corresponding dataloader, it will return (pseudo_label, fname, vid, camid) as a tuple.
    '''
    refined_dataset = copy.deepcopy(dataset) # clone a new dataset
    good_indices = np.argwhere(pseudo_labels != -1).reshape((-1,))
    refined_dataset.train = [refined_dataset.train[i] for i in good_indices]
    
    # TODO keep default order
    default_order = good_indices.astype(np.int).tolist()

    print('>>> Remove {} outliers, re-arrange dataset with {} normal samples.'.format(len(np.argwhere(pseudo_labels==-1).reshape((-1,))), len(good_indices)))
    return RefinedDataset(img_shape, refined_dataset, pseudo_labels[good_indices], default_order, all_train_samples=dataset.train)

def find_latest_checkpoint(ckpts):
    '''
    Find the latest checkpoint file name.

    Args:
        ckpts: list, list of checkpoint names.

    Returns:
        The latest checkpoint file name.
    '''
    ckpts.sort(key=lambda ckpt: int(ckpt.split('-')[-1].split('.')[0]))
    return ckpts[-1]

def opt_to_gpu(opt, is_cuda):
    if not is_cuda:
        return opt
    for state in opt.state.values():
        for k, v, in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    return opt