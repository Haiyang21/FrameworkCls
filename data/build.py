from data.datasets.build import build_dataset
from data.transforms.build import build_transforms
from data.samplers.build import build_sampler
import torch


def build_dataloader(cfg, phase):
    transform = build_transforms(cfg, phase)
    dataset = build_dataset(cfg, phase, transform)
    sampler = build_sampler(cfg, dataset)
    batch_size = cfg.MODEL.BATCH_SIZE
    batch_sampler = None
    if isinstance(sampler, torch.utils.data.sampler.BatchSampler):
        batch_sampler = sampler
        sampler = None
        batch_size = 1

    if phase == 'train':
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            num_workers=cfg.MODEL.WORKERS,
            pin_memory=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.MODEL.BATCH_SIZE,
            num_workers=cfg.MODEL.WORKERS,
            shuffle=False,
            pin_memory=True,
        )

    return data_loader
