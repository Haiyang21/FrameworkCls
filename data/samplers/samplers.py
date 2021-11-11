import torch


def random_sampler(cfg, dataset):
    sampler = torch.utils.data.RandomSampler(dataset)

