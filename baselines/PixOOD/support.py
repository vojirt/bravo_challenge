import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from tqdm import tqdm
from easydict import EasyDict as edict
from sklearn.metrics import roc_curve, auc, average_precision_score

from datasets.bravo import BRAVO


def get_datasets(datasets_folder):
    ['SMIYC', 'ACDC', 'outofcontext', 'synflare', 'synobjs', 'synrain']
    bravo_ACDC_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_ACDC'
    )
    bravo_SMIYC_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_SMIYC'
    )
    bravo_outofcontext_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_outofcontext'
    )
    bravo_synflare_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_synflare'
    )
    bravo_synobjs_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_synobjs'
    )
    bravo_synrain_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_synrain'
    )

    DATASETS = edict(
        bravo_SMIYC=BRAVO(hparams=bravo_SMIYC_config),
        bravo_outofcontext=BRAVO(hparams=bravo_outofcontext_config),
        bravo_synflare=BRAVO(hparams=bravo_synflare_config),
        bravo_synobjs=BRAVO(hparams=bravo_synobjs_config),
        bravo_synrain=BRAVO(hparams=bravo_synrain_config),
        bravo_ACDC=BRAVO(hparams=bravo_ACDC_config)
    )

    return DATASETS

class OODEvaluator:
    def __init__(self, model: nn.Module):
        self.model = model

    def calculate_auroc(self, conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        # print('Started FPR search.')
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        # print(k)
        return roc_auc, fpr_best, k

    def calculate_ood_metrics(self, out, label):
        prc_auc = average_precision_score(label, out)
        roc_auc, fpr, _ = self.calculate_auroc(out, label)
        return roc_auc, prc_auc, fpr

    def evaluate_ood(self, anomaly_score, ood_gts, verbose=True):
        ood_gts = np.concatenate([od.flatten() for od in ood_gts], axis=0)
        anomaly_score = np.concatenate([ass.flatten() for ass in anomaly_score], axis=0)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]

        ood_label = np.ones_like(ood_out, dtype=int)
        ind_label = np.zeros_like(ind_out, dtype=int)

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        if verbose:
            print(f"Calculating Metrics for {len(val_out)} Points ...")

        auroc, aupr, fpr = self.calculate_ood_metrics(val_out, val_label)

        if verbose:
            print(f'Max Logits: AUROC score: {auroc}')
            print(f'Max Logits: AUPRC score: {aupr}')
            print(f'Max Logits: FPR@TPR95: {fpr}')

        result = {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr
        }

        return result

    def compute_anomaly_scores(self, loader, out_path=None, viz_path=None):        
        torch.multiprocessing.set_sharing_strategy('file_system')
        anomaly_score = []
        for x, _, filepath in tqdm(loader, desc="Dataset Iteration"):
            out = self.model.evaluate(x[0])
            savepred = out.pred_y.detach().cpu().numpy().astype(np.uint8).squeeze()

            score = out.anomaly_score.detach().cpu().numpy()
            conf = ((1.0 - score) * 65535).astype(np.uint16)

            destfile = os.path.join(out_path, filepath[0])
            destfile_pred = destfile.replace(loader.dataset.img_suffix, '_pred.png')
            destfile_conf = destfile.replace(loader.dataset.img_suffix, '_conf.png')

            os.makedirs(os.path.dirname(destfile), exist_ok=True)

            # save the prediction
            cv2.imwrite(destfile_pred, savepred)
            cv2.imwrite(destfile_conf, conf)

            if viz_path is not None:
                destfile_segm = os.path.join(viz_path, filepath[0]).replace(loader.dataset.img_suffix, '_segm.jpg')
                os.makedirs(os.path.dirname(destfile_segm), exist_ok=True)
                out.out_segm_img.save(destfile_segm)
                
            anomaly_score.append(score)

        return anomaly_score
