import torch, sys, random, tqdm
import datetime
import logging
import time
from contextlib import contextmanager

sys.path.append('/home/ljc/works/fast-reid')

from models.model import ReidNet
from tools.dataset_wrapper import EvalDataset
from tools.load_config import load_config
from tools.init_gpus import init_gpus
from settings import Settings
from fastreid.evaluation import ReidEvaluator
from fastreid.data.datasets import VeRi, Market1501
from fastreid.utils.logger import log_every_n_seconds


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def inference_on_dataset(model, data_loader, evaluator, gallery_loader):
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    # use gpu
    if torch.cuda.is_available():
        print('>>> Using GPU inference.')
        model = model.cuda()

    with inference_context(model), torch.no_grad():
        for idx, inputs in tqdm.tqdm(enumerate(data_loader), desc='Extract query features'):
            if torch.cuda.is_available():
                inputs['images'] = inputs['images'].cuda()
            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs['images'])
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            idx += 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

        # add gallery data into evaluator
        for idx, inputs in tqdm.tqdm(enumerate(gallery_loader), desc='Extract gallery features'):
            # print('Processing gallery data {}/{} ...'.format(idx+1, len(gallery_loader)))
            if torch.cuda.is_available():
                inputs['images'] = inputs['images'].cuda()
            outputs = model(inputs['images'])
            evaluator.process(inputs, outputs)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device)".format(
            total_time_str, total_time / (total - num_warmup)
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup)
        )
    )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def summary(rank_n, mAP, roc, dataset, num_query, time_cons):
    print('======= Re-id Evaluation ======')
    print('Evaluation configs:\n  - Dataset: {}\n  - Query sample num: {}'.format(dataset, num_query))
    print('Time consumption: {:.3f}s'.format(time_cons))
    print('')
    print('Results:')
    print('')
    print('=== Rank-N Accuracy ===')
    for rn, acc in zip([1,5,10], rank_n):
        print('  - Rank-{} accuracy: {}'.format(rn, acc))
    print('')
    print('=== mAP ===')
    print('mAP: {}'.format(mAP))
    print('')
    print('=== ROC ===')
    for fpr, tpr in zip([1e-4, 1e-3, 1e-2], roc):
        print('  - TPR@FPR={:.0e}: {}'.format(fpr, tpr))
    print('===============================')

def eval_in_training(cfg, model):
    qnum = 1678
    query_dataset = VeRi(root=cfg.DATASET.PATH, mode='query')
    random.shuffle(query_dataset.query)
    query_dataset.query = query_dataset.query[:qnum]
    query_dataset = EvalDataset((256,256), query_dataset, mode='query')

    gallery_dataset = VeRi(root=cfg.DATASET.PATH, mode='gallery')
    gallery_dataset = EvalDataset((256,256), gallery_dataset, mode='gallery')

    q_loader = torch.utils.data.DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=8)
    g_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=8)

    evaluator = ReidEvaluator(cfg, num_query=qnum)

    res = inference_on_dataset(model, q_loader, evaluator, g_loader)

    rank1 = res['Rank-1']
    mAP = res['mAP']

    return mAP, rank1
    

def main():
    Settings.init()
    init_gpus(Settings.gpu_ids)
    cfg = load_config(Settings.conf)
    model = ReidNet(cfg)
    if len(Settings.gpu_ids.split(',')) > 1:
        print('>>> Using multi-GPUs, enable DataParallel')
        model = torch.nn.DataParallel(model)
    print('>>> Using model weights: {}'.format(cfg.TEST.PRETRAINED_MODEL))
    state = torch.load(cfg.TEST.PRETRAINED_MODEL)

    # qnum = 3368
    qnum = 1678 # query num
    # qnum = 100 # query num
    model.load_state_dict(state['model_state_dict'])

    dataset = VeRi(root=cfg.DATASET.PATH, mode='query') # total 1678 query samples
    # dataset = Market1501(root=cfg.DATASET.PATH, mode='query') # 3368 (query)
    random.shuffle(dataset.query)
    dataset.query = dataset.query[:qnum] # choose only qnum samples
    dataset = EvalDataset((256,256), dataset, mode='query')

    gallery_dataset = VeRi(root=cfg.DATASET.PATH, mode='gallery')
    # gallery_dataset = Market1501(root=cfg.DATASET.PATH, mode='gallery')
    gallery_dataset = EvalDataset((256,256), gallery_dataset, mode='gallery')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=8)

    evaluator = ReidEvaluator(cfg, num_query=qnum)
    
    start_time = time.time()
    res = inference_on_dataset(model, dataloader, evaluator, gallery_loader)
    end_time = time.time()

    # unpcaking
    rank_n = [res['Rank-{}'.format(n)] for n in [1,5,10]]
    mAP = res['mAP']
    roc = [res['TPR@FPR={:.0e}'.format(fpr)] for fpr in [1e-4, 1e-3, 1e-2]]

    summary(rank_n, mAP, roc, dataset='VeRi', num_query=qnum, time_cons=(end_time-start_time))

if __name__ == "__main__":
    main()