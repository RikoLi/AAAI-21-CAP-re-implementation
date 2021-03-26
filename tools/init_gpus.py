import os

def init_gpus(gpu_ids):
    '''
    Initialize available GPUs.

    Args:
        gpu_ids: str, available GPU ids.
    '''
    print('>>> Exposing GPU {}'.format(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids