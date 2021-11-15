import os
import logging
import numpy as np
import torch
from collections import defaultdict
from utils.common import is_pytorch_1_1_0_or_later, reduce_loss_dict, reduce_mean, get_rank
from utils.metric_logger import MetricLogger
from utils.eval import cal_precision_and_recall
import torch.distributed as dist


def do_train(
        epoch,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        work_dir,
        distributed,
):
    meters = MetricLogger(delimiter="  ")
    logger = logging.getLogger(__name__)
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for i in range(1, epoch+1):
        model.train()
        if not pytorch_1_1_0_or_later:
            scheduler.step()
        # train
        for batch_i, (imgs, targets, _) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = model(imgs)

            if not isinstance(outputs, list):
                outputs = [outputs]
            loss_dict = dict()
            # process one or multi branch output with the same targets
            for output_id, output in enumerate(outputs):
                loss = criterion(output, targets)

                # if distributed is True:
                #     torch.distributed.barrier()
                #     dist.all_reduce(loss, op=dist.ReduceOp.SUM)

                if isinstance(loss, dict):
                    loss_dict.update(loss)
                else:
                    loss_dict.update({'branch{}'.format(output_id): loss})

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            # meters.update(losses=losses, **loss_dict)
            if batch_i % 50 == 0 or batch_i == len(train_loader) - 1:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        iter=batch_i,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if pytorch_1_1_0_or_later:
            scheduler.step()
        logger.info('[Epoch {}] Training Loss: {}'.format(i, meters))

        # test
        with torch.no_grad():
            pred_labels = defaultdict(list)
            gt_labels = defaultdict(list)

            model.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                outputs = model(imgs.cuda())
                targets = targets.cuda()
                if not isinstance(outputs, list):
                    outputs = [outputs]
                for output_id, output in enumerate(outputs):
                    _, predicts = torch.max(output, 1)

                    pred_labels['branch{}'.format(output_id)].extend(predicts.cpu().tolist())
                    gt_labels['branch{}'.format(output_id)].extend(targets.cpu().tolist())

            for key in sorted(pred_labels):  # branch
                eval_ret, eval_acc = cal_precision_and_recall(gt_labels[key], pred_labels[key])

                logger.info('Validation ' + key)
                for eval_key in eval_ret.keys():
                    logger.info('\t class %d, precision %.3f, recall %.3f, F1 %.3f'
                                % (eval_key, eval_ret[eval_key][0], eval_ret[eval_key][1], eval_ret[eval_key][2]))
                    logger.info('\t accuracy %.3f' % eval_acc)

                    if distributed is True:
                        torch.distributed.barrier()
                        prec = reduce_mean(eval_ret[eval_key][0])
                        recall = reduce_mean(eval_ret[eval_key][1])
                        F1 = reduce_mean(eval_ret[eval_key][2])
                        acc = reduce_mean(eval_acc)
                        logger.info('\t class %d, precision %.3f, recall %.3f, F1 %.3f'
                                    % (eval_key, prec, recall, F1))
                        logger.info('\t accuracy %.3f' % acc)

                if i % 5 == 0 and get_rank() == 0:
                    if not os.path.exists(work_dir):
                        os.makedirs(work_dir)
                    model_path = os.path.join(work_dir, key + "_epoch" + str(i) + "_acc" + str(eval_acc) + ".pth")
                    torch.save(
                        {
                            'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        },
                        model_path
                    )
                    logger.info('Model saved: {}'.format(model_path))
