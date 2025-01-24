import random
import time

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from modules.utils.metric import AverageMeter


def do_train(cfg, model, data_loader, gt, lt, optimizer, criterion, device, logger, epoch, scaler):
    losses = AverageMeter()
    triplet_losses = AverageMeter()
    align_losses = AverageMeter()
    if lt is not None:
        glosses = AverageMeter()
        llosses = AverageMeter()
        alosses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    for idx, batch in enumerate(data_loader):
        x, p, n, a, tp, tn = batch
        n_data = len(x)
        a = a.to(device)
        zeroshot_mode = cfg.MODEL.ATTRIBUTE.MASK_VALUE is not None

        if zeroshot_mode:
            tp = tokenizer(tp, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            tn = tokenizer(tn, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        else:
            rand_num = random.random()
            if rand_num < cfg.INPUT.MISSING_ATTR_PROB:
                a = None
                tp = tokenizer(tp, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                tn = tokenizer(tn, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            elif rand_num < cfg.INPUT.MISSING_ATTR_PROB + cfg.INPUT.MISSING_TEXT_PROB:
                tp = None
                tn = None
            else:
                tp = tokenizer(tp, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                tn = tokenizer(tn, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        gx = torch.stack([gt(i) for i in x], dim=0).to(device)
        gp = torch.stack([gt(i) for i in p], dim=0).to(device)
        gn = torch.stack([gt(i) for i in n], dim=0).to(device)

        data_time.update(time.time() - end)

        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            gx, gx_attnmap = model(gx, a, tp, level='global', mask_value=cfg.MODEL.ATTRIBUTE.MASK_VALUE, zeroshot_mode = zeroshot_mode)
            gp, gp_attnmap = model(gp, a, tp, level='global', mask_value=cfg.MODEL.ATTRIBUTE.MASK_VALUE, zeroshot_mode = zeroshot_mode)
            gn, gn_attnmap = model(gn, a, tn, level='global', mask_value=cfg.MODEL.ATTRIBUTE.MASK_VALUE, zeroshot_mode = zeroshot_mode)

            if a is not None and tp is not None:
                a_embed = model.get_attr_embedding(a)
                t_embed = model.get_text_embedding(tp)
                sim_attr_text = F.cosine_similarity(a_embed, t_embed, dim=1)
                align_loss = torch.mean(torch.clamp(1. - sim_attr_text, min=0))
                align_losses.update(cfg.SOLVER.ALIGN_WEIGHT * align_loss.cpu().item(), n_data)
            else:
                align_loss = torch.tensor(0.0, device=device)
                align_losses.update(0.0, n_data)

            loss = cfg.SOLVER.GLOBAL_WEIGHT * criterion(gx, gp, gn)
            triplet_losses.update(loss.cpu().item(), n_data)
            loss += cfg.SOLVER.ALIGN_WEIGHT * align_loss

            if lt is not None:
                glosses.update(loss.cpu().item(), n_data)

                gx_attnmap = gx_attnmap.cpu().detach().numpy()
                gp_attnmap = gp_attnmap.cpu().detach().numpy()
                gn_attnmap = gn_attnmap.cpu().detach().numpy()
                lx = torch.stack([lt(i, mask) for i, mask in zip(x, gx_attnmap)], dim=0).to(device)
                lp = torch.stack([lt(i, mask) for i, mask in zip(p, gp_attnmap)], dim=0).to(device)
                ln = torch.stack([lt(i, mask) for i, mask in zip(n, gn_attnmap)], dim=0).to(device)

                lx, _ = model(lx, a, level='local')
                lp, _ = model(lp, a, level='local')
                ln, _ = model(ln, a, level='local')

                # local losses
                l = local_loss(criterion, gx, gp, gn, lx, lp, ln)
                llosses.update(cfg.SOLVER.LOCAL_WEIGHT * l[0].cpu().item(), n_data)
                alosses.update(cfg.SOLVER.ALIGN_WEIGHT * l[1].cpu().item(), n_data)
                loss += cfg.SOLVER.LOCAL_WEIGHT * l[0] + cfg.SOLVER.ALIGN_WEIGHT * l[1]

        losses.update(loss.cpu().item(), n_data)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        local_log = (f"Global Loss: {glosses.val:.4f}({glosses.avg:.4f})\t" +
                     f"Local Loss: {llosses.val:.4f}({llosses.avg:.4f})\t" +
                     f"Align Loss: {alosses.val:.4f}({alosses.avg:.4f})\t") if lt is not None else ""
        if idx % cfg.SOLVER.LOG_PERIOD == 0:
            logger.info(f"Train Epoch: [{epoch}][{idx}/{len(data_loader)}]\t" +
                        local_log +
                        f"Loss: {losses.val:.4f}({losses.avg:.4f})\t" +
                        f"Triplet Loss: {triplet_losses.val:.4f}({triplet_losses.avg:.4f})\t" +
                        f"Align Loss: {align_losses.val:.4f}({align_losses.avg:.4f})\t" +
                        f"Batch Time: {batch_time.val:.3f}({batch_time.avg:.3f})\t" +
                        f"Data Time: {data_time.val:.3f}({data_time.avg:.3f})")

    return (losses.avg, glosses.avg, llosses.avg, alosses.avg) if lt is not None else losses.avg

def local_loss(criterion, gx, gp, gn, lx, lp, ln):
    lt_loss = criterion(lx, lp, ln)
    sim_x_ins = F.cosine_similarity(gx, lx, dim=1)
    sim_p_ins = F.cosine_similarity(gp, lp, dim=1)
    sim_n_ins = F.cosine_similarity(gn, ln, dim=1)
    a_loss = torch.mean(torch.clamp(1.-sim_x_ins, min=0)) + \
        torch.mean(torch.clamp(1.-sim_p_ins, min=0)) + \
        torch.mean(torch.clamp(1.-sim_n_ins, min=0))

    return lt_loss, a_loss
