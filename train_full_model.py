import sys

sys.path.append('../..')

import os
import time
import glob
import copy
import torch
import torch.nn as nn
import numpy as np
import tensorboardX

from torch.utils.data.dataloader import DataLoader

from tools.dataset_wrapper import CustomDataset, RefinedDataset
from tools.clusterer import Clusterer
from tools.load_config import load_config
from tools.init_gpus import init_gpus
from tools.split_for_proxy import split_for_proxy
from tools.retrieve_samples import retrieve_all_positive_proxies, retrieve_k_nearest_negative_proxies

from rebuild.train_tools import *
from rebuild.memory_bank import *
from rebuild.sampler import *
from rebuild.dataloaders import *

from models.model import ReidNet

from loss.cam_aware_loss import mat_inter_cam_loss, mat_intra_cam_loss, proxy_loss

from settings import Settings

from fastreid.solver.lr_scheduler import WarmupMultiStepLR
from fastreid.data.datasets import VeRi, Market1501
from fastreid.modeling.backbones.resnet import build_resnet_backbone
from fastreid.solver.optim import Adam
from fastreid.modeling.heads.embedding_head import EmbeddingHead

from evaluate import eval_in_training


def train(cfg, model, dataset, optimizer, scheduler=None, logger=None, is_continue=False, use_pretrained=False, cluster_vis_path=None):
    
    save_to = cfg.TRAIN.CHECKPOINT_PATH
    epochs = cfg.TRAIN.EPOCHS
    batch_size = cfg.TRAIN.BATCHSIZE

    if logger is None:
        print('>>> No tensorboard logger used in training.')
    else:
        print('>>> Logger is used in training.')
        counter = 0

    if len(save_to) == 0:
        print('>>> No checkpoints will be saved.')

    start_ep = 0 # initiate start epoch number
    
    # continue training
    if is_continue:
        print('>>> Continue training from the latest checkpoint.')
        if save_to is None:
            print('>>> Without checkpoint folder, cannot continue training!')
            exit(0)
        ckpts = glob.glob(os.path.join(save_to, '*.pth'))
        if len(ckpts) == 0:
            print('>>> No earlier checkpoints, train from the beginning.')
        else:
            start_ckpt = find_latest_checkpoint(ckpts)
            print('>>> Found earlier checkpoints, continue training with {}.'.format(start_ckpt))

            # load latest model
            start_ep = torch.load(os.path.join(save_to, start_ckpt))['epoch']
            model_state = torch.load(os.path.join(save_to, start_ckpt))['model_state_dict'] # weight, optimizer, scheduler
            opt_state = torch.load(os.path.join(save_to, start_ckpt))['optimizer_state_dict']
            model.load_state_dict(model_state)
            optimizer.load_state_dict(opt_state)
            optimizer = opt_to_gpu(optimizer, torch.cuda.is_available())
            if scheduler is not None:
                scheduler_state = torch.load(os.path.join(save_to, start_ckpt))['scheduler_state_dict']
                scheduler.load_state_dict(scheduler_state)
            if logger is not None:
                counter = torch.load(os.path.join(save_to, start_ckpt))['logger_counter']
    
    # start a new training with pretrained model
    if use_pretrained:
        print('>>> Use pretrained model weights to start a new training.')
        model_state = torch.load(cfg.TRAIN.PRETRAINED_PATH)['model_state_dict'] # weight
        model.load_state_dict(model_state)

    if torch.cuda.is_available():
        model = model.cuda()

    # training loop
    for epoch in range(start_ep, epochs):

        # extract global features
        print('>>> Extracting global features ...')
        global_dataset, features, v_labels, cam_labels = extract_global_features(
            img_shape=(256,256),
            batch_size=batch_size, workers=8,
            model=model, dataset=dataset, is_cuda=torch.cuda.is_available()
        )

        # manually empty the GPU cache
        torch.cuda.empty_cache()

        # clustering
        print('>>> Start clustering ...')
        pseudo_cluster_labels, _, _ = Clusterer(
            features, eps=0.5, is_cuda=torch.cuda.is_available()
        ).cluster(visualize_path=cluster_vis_path, epoch=epoch+1)

        # add new labels
        cam_labels = cam_labels.numpy()
        cam_memory_map = add_new_labels(global_dataset, pseudo_cluster_labels, cam_labels)
        
        # create memory bank
        memory = ProxyMemoryBank(features, global_dataset, momentum=0.2, temp=0.07)
        memory = memory.cuda()

        # training dataset
        train_dataset = TrainDataset((256,256), global_dataset)
        
        # create sampler
        sampler = PBSampler(train_dataset, proxy_num=8, samples_per_proxy=4) # proxy-balanced sampler
        sampler = torch.utils.data.BatchSampler(sampler, batch_size=cfg.TRAIN.BATCHSIZE, drop_last=True)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=8)
        
        # training step
        total_loss = 0
        cnt = 0
        global_label = train_dataset.samples
        for i, (img, vid, camid, batch_cluster_label, proxy_label, abs_proxy_label) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                img = img.cuda()
                vid = vid.cuda()
                camid = camid.cuda()
                batch_cluster_label = batch_cluster_label.cuda()
                proxy_label = proxy_label.cuda()
                abs_proxy_label = abs_proxy_label.cuda()

            # feed into model
            batch_feat = model(img)

            # loss
            loss = memory(batch_feat, abs_proxy_label, camid, batch_cluster_label, epoch, global_label, k=50, inter_loss_epoch=5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log per iteration
            print('epoch: {}/{}, iters: {}/{}, loss: {:.4f}'.format(epoch, epochs, i, len(train_dataloader), loss.item()))
            total_loss += loss.item()
            cnt += 1

        # log per epoch
        logger.add_scalar('loss', total_loss/cnt, global_step=epoch)
        
        if scheduler is not None:
            scheduler.step()
        
        # re-id evaluation per epoch
        mAP, rank1 = eval_in_training(cfg, model)
        print('>>> Epoch: {}, mAP: {:.4f}, rank-1 acc: {:.4f}'.format(epoch, mAP, rank1))
        logger.add_scalar('mAP', mAP, global_step=epoch)
        logger.add_scalar('rank-1_acc', rank1, global_step=epoch)

        # mannully release GPU memory
        del memory, loss, img, vid, camid, batch_cluster_label, proxy_label, abs_proxy_label

        # save checkpoint
        if len(save_to) != 0 and (epoch+1) % cfg.TRAIN.SAVE_INTERVAL == 0:
            save_name = os.path.join(save_to, 'full-model-epoch-{}.pth'.format(epoch+1))
            state_dict = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'logger_counter': counter if logger is not None else None
            }
            torch.save(state_dict, save_name)
            print('>>> Checkpoint is saved as {}.'.format(save_name))



def init_dataset(dataset_name, dataset_root, mode):
    if dataset_name == 'VeRi':
        dataset = VeRi(root=dataset_root, mode=mode)
    elif dataset_name == 'Market1501':
        dataset = Market1501(root=dataset_root, mode=mode)
    else:
        raise ValueError('Wrong dataset name!')
    return dataset

def config_summary(settings, cfg, dataset):
    start_time = time.localtime()
    print('--- Training Configuration Summary ---')
    print('Start time: {}-{}-{} {}:{}:{}'.format(start_time.tm_year, start_time.tm_mon, start_time.tm_mday, start_time.tm_hour, start_time.tm_min, start_time.tm_sec))
    print('Training nums:', len(dataset.train))
    print('Using GPU:', settings.gpu_ids)
    print('Epochs:', cfg.TRAIN.EPOCHS)
    print('Batch size:', cfg.TRAIN.BATCHSIZE)
    print('Learning rate:', cfg.TRAIN.LR)
    print('Weight decay:', cfg.TRAIN.WEIGHT_DECAY)
    print('Checkpoint save interval:', cfg.TRAIN.SAVE_INTERVAL)
    print('Pretrained weight:', cfg.TRAIN.PRETRAINED_PATH)    
    print('--------------------------------------')

def main():
    # load configurations
    Settings.init()
    if Settings.debug:
        print('>>> Debug mode.')
        import ipdb
        ipdb.set_trace()
    init_gpus(Settings.gpu_ids)
    cfg = load_config(Settings.conf)
    dataset = init_dataset(dataset_name='VeRi', dataset_root=cfg.DATASET.PATH, mode='train')
    # dataset.train = dataset.train[:5000] # shorten list for test
    
    # initiate model
    model = ReidNet(cfg)
    model = torch.nn.DataParallel(model)
    # if len(Settings.gpu_ids.split(',')) > 1:
    #     print('>>> Using multi-GPUs, enable DataParallel')
    #     model = torch.nn.DataParallel(model)
    optim = torch.optim.Adam(params=model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = WarmupMultiStepLR(optim, milestones=[20,40], gamma=0.1, warmup_factor=0.01, warmup_iters=10, warmup_method='linear')

    # training monitor
    if len(cfg.TRAIN.LOG_PATH) != 0:
        logger = tensorboardX.SummaryWriter(cfg.TRAIN.LOG_PATH)
    else:
        logger = None


    # print config summary
    config_summary(Settings, cfg, dataset)

    # training
    train(
        cfg, model, dataset, optim, scheduler=scheduler,
        logger=logger, is_continue=Settings.is_continue, use_pretrained=Settings.use_pretrained,
        cluster_vis_path=None
    )

    # close logger
    if logger is not None:
        logger.close()

if __name__ == "__main__":
    main()