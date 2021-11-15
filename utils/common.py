import torch
import logging
import math
import random
import torch
import torch.distributed as dist
from collections import defaultdict


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def reduce_mean(tensor):
    nprocs = get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def is_pytorch_1_1_0_or_later():
    return [int(_) for _ in torch.__version__.split(".")[:3]] >= [1, 1, 0]


def data_distribution(phase, fname, cls_map=None):
    logger = logging.getLogger(__name__)
    if phase not in ['train', 'val']:
        return
    with open(fname) as fp:
        lines = fp.readlines()

    distribution = defaultdict(int)
    for line_id, line in enumerate(lines):
        fn, cls = line.strip().split(' ')
        if cls not in cls_map.keys():
            continue
        distribution[cls] += 1
    logger.info('{} dataset distribution'.format(phase))

    distribution_label_cnt = defaultdict(int)
    distribution_label_cls = dict()
    for key in sorted(distribution):
        distribution_label_cnt[cls_map[key]] += distribution[key]
        if cls_map[key] not in distribution_label_cls.keys():
            distribution_label_cls[cls_map[key]] = list()
        distribution_label_cls[cls_map[key]].append({key: distribution[key]})
    for key in sorted(distribution_label_cnt):
        logger.info('label: {}, count: {}'.format(key, distribution_label_cnt[key]))
        for cls in distribution_label_cls[key]:
            logger.info('\t{}'.format(cls))
    logger.info('--------------------------------------------')



