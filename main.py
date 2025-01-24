import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.config import cfg
from modules.data import build_data
from modules.data.transforms import GlobalTransform, LocalTransform
from modules.engine import do_eval, do_train
from modules.loss import build_loss
from modules.model import build_model
from modules.solver import build_lr_scheduler, build_optimizer
from modules.utils.checkpoint import save_checkpoint
from modules.utils.logger import setup_logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    logger = setup_logger(name=cfg.NAME, level=20, stream=f"{cfg.SAVE_DIR}/{cfg.NAME}/stdout.log")
    logger.info(f"\n{cfg}")

    device = torch.device(cfg.DEVICE)
    model = build_model(cfg)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / all_params) * 100 if all_params > 0 else 0
    logger.info(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percentage:.4f}")
    model.to(device)

    gt = GlobalTransform(cfg)
    lt = LocalTransform(cfg) if cfg.MODEL.LOCAL.ENABLE else None

    if args.test is None:
        train_loader, valid_query_loader, valid_candidate_loader = build_data(cfg)
    else:
        test_query_loader, test_candidate_loader = build_data(cfg, args.test)

    if args.resume is not None:
        path = args.resume
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            logger.info(f"Loaded checkpoint from '{path}'.")
            logger.info(f"Best performance {checkpoint['mAP']} at epoch {checkpoint['epoch']}.")
        else:
            logger.info(f"No checkpoint found at '{path}'.")
            sys.exit()

    # model = torch.compile(model)

    if args.test is not None:
        logger.info(f"Begin test on {args.test} set.")
        do_eval(
            model,
            test_query_loader,
            test_candidate_loader,
            gt,
            lt,
            cfg.DATA.ATTRIBUTES.NAME,
            device,
            logger,
            cfg.SOLVER.BETA,
            missing_attr_prob=0.0,
            missing_text_prob=0.0
        )
        sys.exit()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    criterion = build_loss(cfg)

    best_mAP = 0
    start = time.time()
    scaler = torch.amp.GradScaler(device=device)
    tbwriter = SummaryWriter(log_dir=os.path.join(cfg.SAVE_DIR, cfg.NAME, 'tensorboard_logs'))
    for epoch in range(cfg.SOLVER.EPOCHS):
        logger.info(f"Global branch learning rate: {optimizer.param_groups[0]['lr']}.")
        if cfg.MODEL.LOCAL.ENABLE:
            logger.info(f"Local branch learning rate: {optimizer.param_groups[2]['lr']}.")

        losses = do_train(
            cfg,
            model,
            train_loader,
            gt,
            lt,
            optimizer,
            criterion,
            device,
            logger,
            epoch+1,
            scaler
        )
        if cfg.MODEL.LOCAL.ENABLE is False:
            tbwriter.add_scalar('train/losses', losses, epoch+1)
        else:
            tbwriter.add_scalar('train/losses', losses[0], epoch+1)
            tbwriter.add_scalar('train/global', losses[1], epoch+1)
            tbwriter.add_scalar('train/local', losses[2], epoch+1)
            tbwriter.add_scalar('train/align', losses[3], epoch+1)

        if (epoch+1) % cfg.SOLVER.EVAL_STEPS == 0:
            mAP = do_eval(
                model,
                valid_query_loader,
                valid_candidate_loader,
                gt,
                lt,
                cfg.DATA.ATTRIBUTES.NAME,
                device,
                logger,
                cfg.SOLVER.BETA,
                missing_attr_prob=0.0,
                missing_text_prob=0.0
            )
            tbwriter.add_scalar('valid/mAP', mAP, epoch+1)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'epoch': epoch+1,
                'model': model.state_dict(),
                'mAP': mAP
            }, is_best, path=os.path.join(cfg.SAVE_DIR, cfg.NAME))

        scheduler.step()

    tbwriter.close()
    end = time.time()
    total_time = end - start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total training time: {total_time_str}')

    best_model_path = os.path.join(cfg.SAVE_DIR, cfg.NAME, 'model_best.pth.tar')
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded best model from '{best_model_path}' at epoch {checkpoint['epoch']} with mAP {checkpoint['mAP']}.")
    else:
        logger.info(f"No best model found at '{best_model_path}'.")
        sys.exit()
    
    test_query_loader, test_candidate_loader = build_data(cfg, 'TEST')

    logger.info("Testing with full modalities (missing_attr_prob=0.0, missing_text_prob=0.0)")
    do_eval(
        model,
        test_query_loader,
        test_candidate_loader,
        gt,
        lt,
        cfg.DATA.ATTRIBUTES.NAME,
        device,
        logger,
        cfg.SOLVER.BETA,
        missing_attr_prob=0.0,
        missing_text_prob=0.0
    )

    logger.info("Testing with missing text only (missing_attr_prob=0.0, missing_text_prob=1.0)")
    do_eval(
        model,
        test_query_loader,
        test_candidate_loader,
        gt,
        lt,
        cfg.DATA.ATTRIBUTES.NAME,
        device,
        logger,
        cfg.SOLVER.BETA,
        missing_attr_prob=0.0,
        missing_text_prob=1.0
    )

    logger.info("Testing with missing attribute only (missing_attr_prob=1.0, missing_text_prob=0.0)")
    do_eval(
        model,
        test_query_loader,
        test_candidate_loader,
        gt,
        lt,
        cfg.DATA.ATTRIBUTES.NAME,
        device,
        logger,
        cfg.SOLVER.BETA,
        missing_attr_prob=1.0,
        missing_text_prob=0.0
    )


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument("--cfg", nargs="+", help="config file", default=None, type=str)
    parser.add_argument("--test", help="run test on validation or test set", default=None, type=str)
    parser.add_argument("--resume", help="checkpoint model to resume", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(1)  # Solve CPU utilization problem
    args = parse_args()
    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()
    set_seed()
    main(cfg)
