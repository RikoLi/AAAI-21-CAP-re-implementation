This project is a re-implementation work of AAAI 2021 paper *Camera-aware Proxies for Unsupervised Person Re-ID*, with the dataset changed to [VeRi](https://vehiclereid.github.io/VeRi/).

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

Baseline model has been implemented.

## Re-id Performance

The model is evaluated on 1 GTX1080 GPU with all query and gallery data in VeRi. Evaluation tools are from fast-reid tool box.

Performance on other datasets may be evaluated in the future.

![baseline_performance](./baseline_performance.png)
