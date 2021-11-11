from data.samplers.samplers import random_sampler


META_SAMPLER = {
    'random_sampler': random_sampler,
}


def build_sampler(cfg, dataset):
    meta_sampler = META_SAMPLER[cfg.DATASET.SAMPLER]
    return meta_sampler(cfg, dataset)
