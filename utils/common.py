import torch
import logging
import math
import random
from collections import defaultdict


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



