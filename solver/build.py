import torch


def make_optimizer(cfg, params):
    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise ValueError('Optimizer not supported')

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.STEP_SIZE, gamma=0.1)
    elif cfg.SOLVER.SCHEDULER == 'exponential_lr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif cfg.SOLVER.SCHEDULER == 'cosine_annealing_lr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.T_MAX, eta_min=0)
    else:
        raise ValueError('Scheduler not supported')

    return scheduler

