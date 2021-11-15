import os
import torch
from config.defaults import _C as cfg
from utils.logger import setup_logger
from modeling.build import build_model
from loss.build import build_loss
from data.build import build_dataloader
from solver.build import make_optimizer, make_lr_scheduler
from trainer import do_train
from utils.checkpoint import ClassificationCheckpointer
import argparse
import torch.distributed as dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Classification Training")
    parser.add_argument(
        "--config-file",
        default="config/cls.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        cfg.DISTRIBUTED = True
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # output
    work_dir = cfg.OUTPUT.WORK_DIR
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # log
    logger = setup_logger('', work_dir, get_rank())  # setup_logger中的name参数要设置成''，否则，其他文件中的logger无法输出信息
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("running with config:\n{}".format(cfg))

    # model
    local_rank = args.local_rank
    model = build_model(cfg)
    model.to(torch.device("cuda"))
    if cfg.DISTRIBUTED is True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    logger.info("running with model:\n{}".format(model))

    # optimizer
    optimizer = make_optimizer(cfg, model.parameters())
    scheduler = make_lr_scheduler(cfg, optimizer)

    # load weight
    # save_to_disk = get_rank() == 0
    # checkpointer = ClassificationCheckpointer(
    #     cfg, model, optimizer, scheduler, work_dir, save_to_disk
    # )
    # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)

    # criterion
    criterion = build_loss(cfg)
    criterion = criterion.cuda(local_rank)

    # data
    train_dataloader = build_dataloader(cfg, 'train')
    val_dataloader = build_dataloader(cfg, 'val')

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
        work_dir=work_dir,
        distributed=cfg.DISTRIBUTED,
    )
