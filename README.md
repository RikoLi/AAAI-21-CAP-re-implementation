# CAP Re-implementation

This project is a re-implementation work of AAAI 2021 paper *Camera-aware Proxies for Unsupervised Person Re-ID*, with the dataset changed to [VeRi](https://vehiclereid.github.io/VeRi/). 

The official implementation version is open-sourced at [CAP-master](https://github.com/Terminator8758/CAP-master
).

## Prerequisites

This work is partially based on [fast-reid toolbox](https://github.com/JDAI-CV/fast-reid/tree/master) proposed by JDAI-CV. Running environment configurations are showed on the github page of fast-reid.

## TODOs

Current branch: baseline

- [x] Model defination
- [x] Settings and configurations
- [x] Pseudo cluster label
- [x] Memory bank
- [x] Cluster-balanced sampling
- [x] Warm-up LR scheduler

For full model with intra/inter-camera loss, check for [final model branch](https://github.com/RikoLi/camera-aware-re-implementation/tree/final-model).

## Re-id Performance

The model is evaluated on 1 GTX1080 GPU with all query and gallery data in VeRi. Evaluation tools are from fast-reid tool box. Performance on other datasets may be evaluated in the future.


Model | mAP | Rank-1 Accuracy
--    | --  | --
baseline | 28.5% | 65.7%
intra-cam loss only | 43.6% | 83.3%
final model (this work) | **44.5%** | **88.3%**
final model (official) | 40.6% | 87.0%

- baseline: The performance of the pure clustering-based method for pseudo label assignment with cluster-balanced sampling.
- intra-cam loss only: The performance of the method using only intra-camera loss and proxy-balanced sampling.
- final model (this work): The performance of the re-implementation work using both intra-camera and inter-camera loss and proxy-balanced sampling.
- final model (official): The performance of the official implementation of the final model.

## Usage

### Dataset

In this re-implementation works, the model was only trained and evaluated on VeRi dataset. Dataset can be found at [VeRi](https://vehiclereid.github.io/VeRi/). The usage of dataset needs authorization.

The folder of the dataset should be placed under `DATASET.PATH`. For example, the unzipped dataset folder `VeRi/` should be placed as `/home/somebody/datasets/VeRi/` for the setting `DATASET.PATH="/home/somebody/datasets"`.

### Configurations

Edit `configs/baseline_conf.yml` for your own training/evaluating settings. The format of the configuration file follows the requirement of the fast-reid toolbox. Here goes a simple introduction.

```yml
# Resnet50 model for VeRi dataset

# model settings
MODEL:
  BACKBONE:
    PRETRAIN: True
    PRETRAIN_PATH: False
    LAST_STRIDE: 1
    NORM: "BN"
    WITH_IBN: False
    WITH_SE: False
    WITH_NL: False
    DEPTH: "50x"
    FEAT_DIM: 2048

# dataset settings
DATASET:
  PATH: "/home/ljc/datasets" # Change to your own dataset path

# training settings
TRAIN:
  EPOCHS: 50		# Total training epochs
  BATCHSIZE: 32		# Batch size of each iteration
  LR: 0.00035		# Learning rate
  WEIGHT_DECAY: 0.0005	# Weight decay factor
  SAVE_INTERVAL: 10	# Checkpoint saving interval
  PRETRAINED_PATH: ""	# Leave it empty if you don't start from a pretrained model
  CHECKPOINT_PATH: ""	# Leave it empty if you don't need to save checkpoints
  LOG_PATH: ""		# Leave it empty if you don't need a training log
  CLUSTER_VIS_PATH: ""	# Leave it empty if you don't need to visualize clustering results

# evaluation settings
TEST:
  PRETRAINED_MODEL: "/home/somebody/my_model.pth" # Change to your own model
  AQE:
    ENABLED: True
    QE_TIME: 1
    QE_K: 10
    ALPHA: 3.0
  METRIC: "cosine"
  # default re-ranking configs
  RERANK:
    ENABLED: False
    K1: 20
    K2: 6
    LAMBDA: 0.3
  ROC_ENABLED: True
```

For more details about the config file, please check fast-reid docs.

### Training

Start training with the shell script.

```bash
./start_train.sh
```
Or you can use other configs and GPU assignments by

```bash
python train.py --conf ./my_config.yml --gpu_ids 0,1,2,3 # seperate with "," when using more than one GPU
```

You can continue training from the **latest** checkpoint by adding `--is_continue` or edit and run `continue_train.sh`. See `settings.py` for argument details.

### Evaluation

Start evaluation with the shell script.

```bash
./start_eval.sh
```
