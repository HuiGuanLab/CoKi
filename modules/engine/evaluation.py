import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPTokenizer

from modules.utils.metric import APScorer, AverageMeter


def do_eval(model, query_loader, candidate_loader, gt, lt, attrs, device, logger, beta, missing_attr_prob, missing_text_prob):
    mAPs = AverageMeter()

    logger.info("Begin evaluation.")
    model.eval()

    logger.info("Forwarding query images...")
    q_feats, q_values = extract_features(model, query_loader, gt, lt, device, len(attrs), beta, missing_attr_prob, missing_text_prob)
    logger.info("Forwarding candidate images...")
    c_feats, c_values = extract_features(model, candidate_loader, gt, lt, device, len(attrs), beta, missing_attr_prob, missing_text_prob)

    for i, attr in enumerate(attrs):
        mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
        logger.info(f"{attr} MeanAP: {100.*mAP:.4f}")
        mAPs.update(mAP, q_feats[i].shape[0])

    logger.info(f"Total MeanAP: {100.*mAPs.avg:.4f}")

    return mAPs.avg


def extract_features(model, data_loader, gt, lt, device, n_attrs, beta, missing_attr_prob, missing_text_prob):
    feats = []
    indices = [[] for _ in range(n_attrs)]
    values = []
    with tqdm(total=len(data_loader), file=sys.stderr) as bar:
        cnt = 0
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        for idx, batch in enumerate(data_loader):
            x, a, v, t = batch
            a = a.to(device)
            # t = tokenizer(t, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            a_processed, t_processed = handle_missing_data(a, t, device, tokenizer, missing_attr_prob, missing_text_prob)

            out = process_batch(model, x, a_processed, t_processed, gt, lt, device, beta)
            feats.append(out.cpu().numpy())
            values.append(v.numpy())

            for i in range(a.size(0)):
                indices[a[i].cpu().item()].append(cnt)
                cnt += 1

            bar.update(1)

    feats = np.concatenate(feats)
    values = np.concatenate(values)

    feats = [feats[indices[i]] for i in range(n_attrs)]
    values = [values[indices[i]] for i in range(n_attrs)]

    return feats, values


def handle_missing_data(a, t, device, tokenizer, missing_attr_prob, missing_text_prob):
    # 初始化处理后的属性和文本
    a_processed = a.clone().to(device)
    t_processed = tokenizer(t, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    rand_num = random.random()

    if rand_num < missing_attr_prob:
        # 只有属性缺失
        a_processed = None
    elif rand_num < missing_attr_prob + missing_text_prob:
        # 只有文本缺失
        t_processed = None
        # t = [""] * len(t)
        # t_processed = tokenizer(t, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    else:
        # 属性和文本都存在，保持 a_processed 和 t_processed 不变
        pass

    return a_processed, t_processed


def process_batch(model, x, a, t, gt, lt, device, beta):
    gx = torch.stack([gt(i) for i in x], dim=0)
    gx = gx.to(device)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
        g_feats, attmap = model(gx, a, t, level='global')

    if lt is None:
        return nn.functional.normalize(g_feats, p=2, dim=1)

    attmap = attmap.cpu().numpy()
    lx = torch.stack([lt(i, mask) for i, mask in zip(x, attmap)], dim=0)
    lx = lx.to(device)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
        l_feats, _ = model(lx, a, level='local')

    out = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feats, p=2, dim=1),
                    torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feats, p=2, dim=1)), dim=1)

    return out


def mean_average_precision(queries, candidates, q_values, c_values):
    '''
    calculate mAP of a conditional set. Samples in candidate and query set are of the same condition.
        cand_set: 
            type:   nparray
            shape:  c x feature dimension
        queries:
            type:   nparray
            shape:  q x feature dimension
        c_gdtruth:
            type:   nparray
            shape:  c
        q_gdtruth:
            type:   nparray
            shape:  q
    '''

    scorer = APScorer(candidates.shape[0])

    # similarity matrix
    simmat = np.matmul(queries, candidates.T)

    ap_sum = 0
    for q in range(simmat.shape[0]):
        sim = simmat[q]
        index = np.argsort(sim)[::-1]
        sorted_labels = []
        for i in range(index.shape[0]):
            if c_values[index[i]] == q_values[q]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)

        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP = ap_sum / simmat.shape[0]

    return mAP
