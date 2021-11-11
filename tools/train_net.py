import os
import torch
from config.defaults import _C as cfg
from utils.logger import setup_logger
from modeling.build import build_model
from loss.build import build_loss
from data.build import build_dataloader
from solver.build import make_optimizer, make_lr_scheduler
from trainer import do_train


if __name__ == "__main__":
    # config
    config_file = 'config/cls.yaml'
    cfg.merge_from_file(config_file)
    cfg.freeze()

    # output
    work_dir = cfg.OUTPUT.WORK_DIR
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # log
    logger = setup_logger('', work_dir, 0)  # setup_logger中的name参数要设置成''，否则，其他文件中的logger无法输出信息
    logger.info("loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("running with config:\n{}".format(cfg))

    # model
    model = build_model(cfg)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    logger.info("running with model:\n{}".format(model))

    # criterion
    criterion = build_loss(cfg)

    # data
    train_dataloader = build_dataloader(cfg, 'train')
    val_dataloader = build_dataloader(cfg, 'val')

    # optimizer
    optimizer = make_optimizer(cfg, model.parameters())
    scheduler = make_lr_scheduler(cfg, optimizer)

    # train
    logger.info("start training")
    logger.info("train_loader: {}".format(len(train_dataloader)))
    logger.info("val_loader: {}".format(len(val_dataloader)))
    do_train(
        epoch=cfg.SOLVER.MAX_EPOCH,
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        work_dir=work_dir
    )
