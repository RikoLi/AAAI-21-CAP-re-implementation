In this branch, the full model trained with intra/inter-camera loss is completed.

## TODOs

- [x] Proxy label assignment
- [x] Project rebuild
- [x] Proxy memory bank
- [x] Proxy-balanced sampling
- [x] Intra-camera loss
- [x] Inter-camera loss

## Usage

### Training

Similar to the steps in baseline model training. Start training with the shell script.

```bash
./start_train.sh
```

Or you can directly start it with python by

```bash
python train_full_model.py --conf ./my_config.yml --gpu_ids 0,1,2,3
```

### Evaluation

Start evaluation with the shell script.

```bash
./start_eval.sh
```

