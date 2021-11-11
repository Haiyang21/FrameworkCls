import torch
from loss.focal_loss import FocalLoss


def cross_entropy_loss(cfg):
    return torch.nn.CrossEntropyLoss()


def focal_loss(cfg):
    return FocalLoss(gamma=2)


META_LOSS = {
    'focal_loss': focal_loss,
    'cross_entropy_loss': cross_entropy_loss,
}


def build_loss(cfg, loss_name=None):
    if loss_name is None:
        loss_name = cfg.SOLVER.LOSS_NAME
    meta_loss = META_LOSS[loss_name]
    return meta_loss(cfg)
