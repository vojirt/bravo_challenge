import torch
import os
from torch import nn
from types import SimpleNamespace
from einops import rearrange
from tqdm import tqdm
import importlib
import numpy as np

from helpers.saver import Saver
from helpers.kmeans import (
    OnlineKMeans,
    DiffKMeansMultiClassv2,
    majority_vote_patch_label_selection,
    patch_label_distribution,
)
from helpers.neyman_pearson import (
    estimate_neyman_pearson_task,
    eval_neyman_pearson_task,
)

class CacheLoader():
    def __init__(self, dataloader, outdir, strhash, model, device, emb_size):
        self.model = model
        self.device = device
        self.out_dir = os.path.join(outdir, strhash)
        os.makedirs(self.out_dir, exist_ok=True)
        self.indexes = np.arange(len(dataloader))
        
        print(f"Caching data to {self.out_dir}") 
        first_id, last_id = 0, len(dataloader) - 1
        if not (os.path.isfile(os.path.join(self.out_dir, f"{first_id:08d}.pt")) and 
                os.path.isfile(os.path.join(self.out_dir, f"{last_id:08d}.pt"))):
            for i, sample in enumerate(tqdm(dataloader)):
                filename = os.path.join(self.out_dir, f"{i:08d}.pt")
                # [B, 3, H, W], [B, H, W]
                image, target = sample[0].to(self.device), sample[1].to(self.device)
                out = self.model(image) 
                emb = out.emb[:, :, :, -emb_size :]
                torch.save({"target":target.cpu(), "emb":emb.cpu()}, filename)
        else:
            print(f"    caching data exist, skipping precomputing.") 

        self.current = -1 

    def shuffle(self):
        self.indexes = np.random.permutation(self.indexes)

    def __len__(self):
        return self.indexes.shape[0]

    def get_id(self, index):
        if index >= self.indexes.shape[0]:
            raise IndexError

        filename = os.path.join(self.out_dir, f"{self.indexes[index]:08d}.pt")
        sample = torch.load(filename, torch.device('cpu'))
        target = sample["target"].to(self.device)
        emb = sample["emb"].to(self.device)
            
        return {"target":target, "emb": emb}


class GROODNet(nn.Module):
    def __init__(self, cfg):
        super(GROODNet, self).__init__()
        self.NUM_CLASSES = cfg.MODEL.NUM_CLASSES
        self.EMB_SIZE = cfg.MODEL.EMB_SIZE
        self.PATCH_SIZE = cfg.MODEL.PATCH_SIZE
        self.MAJOR_VOTE_THR = cfg.MODEL.PATCH_MAJORITY_VOTE_THR
        self.IGNORE_LABEL = cfg.LOSS.IGNORE_LABEL
        self.mahalanobis = cfg.MODEL.MAHALANOBIS_DIST
        self.best_checkpoint = cfg.MODEL.BEST_BACKBONE_CHECKPOINT

        self.custom_data = {}
        self._dist = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_module = importlib.import_module("net.models." + cfg.MODEL.BACKBONE_FILENAME)
        self.model = getattr(model_module, cfg.MODEL.BACKBONE_NET)(**{"cfg": cfg})

        if os.path.isfile(cfg.MODEL.BACKBONE_CHECKPOINT):
            Saver.load_checkpoint(cfg.MODEL.BACKBONE_CHECKPOINT, self.model, device=self.device)
        else:
            raise RuntimeError(f"\033[31m No checkpoint found for the backbone/LP model!\033[39m")

        # froze parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.feature_resize_factor = cfg.MODEL.FEATURE_RESIZE_FACTOR

        def dist2sim_(dists, slope=0.5):
            return 1.0 / (1.0 + dists * slope)

        self.dist2sim = dist2sim_


    def compute_2dspace_vectors(self, out, eval_scale_factor=1):
        if eval_scale_factor > 1: 
            emb = rearrange( out.emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> b c hp wp")
            logits = rearrange(out.logits_embshape, "b hp wp c -> b c hp wp")

            emb = torch.nn.functional.interpolate(emb, scale_factor=eval_scale_factor, mode="bilinear")
            logits = torch.nn.functional.interpolate(logits, scale_factor=eval_scale_factor, mode="bilinear")

            emb = rearrange(emb, "b c hp wp -> (b hp wp) c")
            logits = rearrange(logits, "b c hp wp -> (b hp wp) c")
        else:
            # [B * Hp * Wp, C]
            emb = rearrange(out.emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> (b hp wp) c")
            logits = rearrange(out.logits_embshape, "b hp wp c -> (b hp wp) c")

        nm_dist = torch.zeros_like(logits)
        for c in range(0, self.NUM_CLASSES):
            nm_dist[:, c] = self._dist(emb, class_id=c)
        return logits, nm_dist

    def preprocess_target(self, target):
        target, target_weights = majority_vote_patch_label_selection(target, patch_size=self.PATCH_SIZE // self.feature_resize_factor)
        target[target_weights < self.MAJOR_VOTE_THR] = self.IGNORE_LABEL
        return target

    def train_n_p_task(self, trainer):
        logits_list = []
        nm_dist_list = []
        labels_list = []
        print("Extracting logits and nm_dist for N-P task")
        for sample in tqdm(trainer.train_loader):
            # [B, 3, H, W], [B, H, W]
            image, target = sample[0].to(self.device), sample[1]
            # [B * Hp * Wp]
            labels_list.append(self.preprocess_target(target))

            out = self.model(image)
            # [B * Hp * Wp, C]
            logits, nm_dist = self.compute_2dspace_vectors(out)
            logits_list.append(logits.cpu())
            nm_dist_list.append(nm_dist.cpu())

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        nm_dist = torch.cat(nm_dist_list, dim=0)

        n_p_model = estimate_neyman_pearson_task(labels, logits, nm_dist, self.NUM_CLASSES, dist2sim=self.dist2sim)

        self.custom_data["n_p_model"] = n_p_model

        logits_list = []
        nm_dist_list = []
        labels_list = []
        print("Extracting logits and nm_dist for test of N-P task")
        for sample in tqdm(trainer.val_loader):
            # [B, 3, H, W], [B, H, W]
            image, target = sample[0].to(self.device), sample[1]
            # [B * Hp * Wp]
            labels_list.append(self.preprocess_target(target))

            out = self.model(image)
            # [B * Hp * Wp, C]
            logits, nm_dist = self.compute_2dspace_vectors(out)
            logits_list.append(logits.cpu())
            nm_dist_list.append(nm_dist.cpu())

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        nm_dist = torch.cat(nm_dist_list, dim=0)

        score = eval_neyman_pearson_task(n_p_model, logits, nm_dist, self.NUM_CLASSES)
        y_pred = torch.argmax(score, dim=-1)

        cls_acc = []
        for c in torch.unique(labels):
            if c == self.IGNORE_LABEL:
                continue
            yc_true = labels == c
            yc_pred = y_pred == c
            sum_c = torch.sum(yc_true).item()
            cls_acc.append(torch.sum(yc_true & yc_pred).item() / sum_c)
        print(f"Mean per-class accuracy GROOD N-P score: {np.mean(cls_acc)*100:0.2f}")

    def forward(self, x, eval_scale_factor=1):
        with torch.no_grad():
            out = self.model(x)
            logits, nm_dist = self.compute_2dspace_vectors(out, eval_scale_factor=eval_scale_factor)
            score = eval_neyman_pearson_task(
                self.custom_data["n_p_model"], logits, nm_dist, self.NUM_CLASSES
            )

        pred_y = torch.argmax(logits, dim=-1)
        pred_score = score[torch.arange(score.shape[0]), pred_y]

        pred_score = rearrange( pred_score, "(b h w) -> b h w", b=x.shape[0], h=eval_scale_factor*out.emb.shape[1], w=eval_scale_factor*out.emb.shape[2])
        pred_score_all = rearrange(score, "(b h w) c -> b h w c", b=x.shape[0], h=eval_scale_factor*out.emb.shape[1], w=eval_scale_factor*out.emb.shape[2])
        pred_y = rearrange( pred_y.float(), "(b h w) -> b h w", b=x.shape[0], h=eval_scale_factor*out.emb.shape[1], w=eval_scale_factor*out.emb.shape[2])
        nm_dist = rearrange( nm_dist, "(b h w) c -> b h w c", b=x.shape[0], h=eval_scale_factor*out.emb.shape[1], w=eval_scale_factor*out.emb.shape[2])
        logits = rearrange( logits, "(b h w) c -> b h w c", b=x.shape[0], h=eval_scale_factor*out.emb.shape[1], w=eval_scale_factor*out.emb.shape[2])

        return SimpleNamespace(pred_y=pred_y, pred_score=pred_score, pred_score_all=pred_score_all, nm_dist=nm_dist, logits=logits)

    def nm_model_accuracy(self, trainer, on_train_data=True):
        y_pred = []
        y_pred_lp = []
        y_true = []
        tbar = tqdm(trainer.train_loader) if on_train_data else tqdm(trainer.val_loader)

        for sample in tbar:
            # [B, 3, H, W], [B, H, W]
            image, target = sample[0].to(self.device), sample[1].to(self.device)
            # [B * Hp * Wp, C]
            out = self.model(image)
            emb = rearrange( out.emb[:, :, :, -self.EMB_SIZE:], "b hp wp c -> (b hp wp) c")

            # [B * Hp * Wp]
            target = self.preprocess_target(target)

            scores = torch.zeros(size=[target.shape[0], self.NUM_CLASSES])
            for c in range(0, self.NUM_CLASSES):
                scores[:, c] = -self._dist(emb, class_id=c)

            y_pred.extend(torch.argmax(scores, dim=-1).flatten().cpu().numpy())
            y_true.extend(target.cpu().numpy())

            y_pred_lp.extend(
                torch.argmax(out.logits_embshape, dim=-1).flatten().cpu().numpy()
            )

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_lp = np.array(y_pred_lp)
        cls_acc = []
        cls_acc_lp = []
        for c in np.unique(y_true):
            if c == self.IGNORE_LABEL:
                continue
            yc_true = y_true == c
            yc_pred = y_pred == c
            yc_pred_lp = y_pred_lp == c
            sum_c = float(np.sum(yc_true))
            cls_acc.append(np.sum(yc_true & yc_pred) / sum_c)
            cls_acc_lp.append(np.sum(yc_true & yc_pred_lp) / sum_c)
        print(f"Mean per-class accuracy NM: {np.mean(cls_acc)*100:0.2f}, LP: {np.mean(cls_acc_lp)*100:0.2f}")
        return np.mean(cls_acc)

    def nm_model_exists(self):
        raise NotImplementedError

    def compute_nm_model(self):
        raise NotImplementedError

    def post_training(self, trainer):
        if self.RECOMPUTE_NM or not self.nm_model_exists(): 
            self.compute_nm_model(trainer) 
        else:
            print(f"Skipping computation of NM, RECOMPUTE_NM: {self.RECOMPUTE_NM}, knn_model exists:", self.nm_model_exists())

        self.train_n_p_task(trainer)

class GROODNetKNM(GROODNet):
    def __init__(self, cfg):
        super(GROODNetKNM, self).__init__(cfg)
        self.NUM_KNN_ITERATIONS = 5 
        self.NUM_KNN_TRIALS = 5 
        self.RECOMPUTE_NM = cfg.EXPERIMENT.RECOMPUTE_NM
        self._dist = self.knn_dist_min
        
    def knn_dist(self, emb, class_id, k_cluster=None):
        if k_cluster is None:
            k_cluster = self.custom_data["knn_model"][class_id]
        diff = []
        for k in range(0, k_cluster.clusters.shape[0]):
            d = k_cluster._dist_fnc(
                emb,
                k_cluster.clusters[k : k + 1, :].to(self.device),
                cov=k_cluster.cluster_var[k, ...].to(self.device),
            )
            diff.append(d)
        diff = torch.cat(diff, dim=-1)
        return diff

    def knn_dist_min(self, emb, class_id):
        return torch.min(self.knn_dist(emb, class_id), dim=-1)[0]

    def nm_model_exists(self):
        return "knn_model" in self.custom_data

    def compute_nm_model(self, trainer):
        accuracy = []
        knn_models = []
        all_stats = []
        for i in range(0, self.NUM_KNN_TRIALS):
            print(f"--------------------------------------------")
            print(f"Running KNN fitting, trial {i}.")
            knn_models.append(self.fit_nm_model(trainer))
            # set model
            self.custom_data["knn_model"] = knn_models[i]
            print(f"Running KNN model accuracy testing, trial {i}.")
            # accuracy.append(self.nm_model_accuracy(trainer, on_train_data=False))
            P, cls_acc, cls_prob, prob = self.nm_clusters_accuracy(knn_models[i], trainer, on_train_data=False, full=True) 
            print(f"Cluster P: {P*100:0.2f}; Mean acc.: {cls_acc*100:0.2f}, Mean prob. acc: {cls_prob*100:0.2f}, log prob: {prob:0.3f}")
            accuracy.append(cls_acc)
            all_stats.append([P, cls_acc, cls_prob, prob])

        self.custom_data["all_knn_models"] = knn_models
        self.custom_data["all_knn_models_stats"] = np.array(all_stats)
        self.custom_data["knn_model"] = knn_models[np.argmax(accuracy)]
        print(f"Best KNN model mean class accuracy {np.max(accuracy)*100:0.2f}%")

    def fit_nm_model(self, trainer):
        # dataloader = CacheLoader(trainer.train_loader, os.path.join(trainer.cfg.EXPERIMENT.OUT_DIR, trainer.cfg.EXPERIMENT.NAME, "cache_knn"), 
        dataloader = CacheLoader(trainer.train_loader, os.path.join(trainer.cfg.MODEL.BACKBONE_EXP_DIR, "cache_emb"), 
                                "train_" + trainer.cfg.DATASET.TRAIN, self.model, self.device, self.EMB_SIZE)
        class_clusters = [
            OnlineKMeans(mahalanobis_dist=self.mahalanobis)
            for _ in range(0, self.NUM_CLASSES)
        ]
        print("Training K-NN model ...")
        for it in range(0, self.NUM_KNN_ITERATIONS):
            print(f"=== Iteration {it} ===")
            # for sample in tqdm(trainer.train_loader):
            dataloader.shuffle()
            for sample_id in tqdm(range(0, len(dataloader))):
                # # [B, 3, H, W], [B, H, W]
                # image, target = sample[0].to(self.device), sample[1].to(self.device)
                # # [B * Hp * Wp, C]
                # emb = rearrange(
                #     self.model(image).emb[:, :, :, -self.EMB_SIZE :],
                #     "b hp wp c -> (b hp wp) c",
                # )
                sample = dataloader.get_id(sample_id)
                target, emb = sample["target"], sample["emb"]
                # [B * Hp * Wp, C]
                emb = rearrange(emb, "b hp wp c -> (b hp wp) c")

                # [B * Hp * Wp]
                target = self.preprocess_target(target)

                for ul in torch.unique(target):
                    if ul == self.IGNORE_LABEL:
                        continue
                    mask = target == ul
                    class_clusters[ul].add_batch(emb[mask, :])

            if (it > 0 and it % 2 == 0) or it == self.NUM_KNN_ITERATIONS - 1:
                # P = self.nm_clusters_accuracy(
                #     class_clusters, trainer, on_train_data=True
                # )
                for ci, k in enumerate(class_clusters):
                #     # remove clusters that has less then 50% accuracy,
                #     # i.e. more than 50% of closest training data assigned to this cluster are from incorrect classes
                #     to_remove = (
                #         torch.nonzero((P[ci] <= 0.5).logical_or(torch.isnan(P[ci])))[:, 0]
                #         .cpu()
                #         .numpy()
                #     )
                #     if len(to_remove) > 0 and k.count > 1 and len(to_remove) < k.count:
                #         k.remove_clusters(to_remove)
                    # remove "bad" clusters
                    k.consolidate()
            string = "Cluster Counts: \n"
            for j in range(0, len(class_clusters)):
                string += f"K{j}={class_clusters[j].count:d}  "
            print(string)
        return class_clusters

    def nm_clusters_accuracy(self, class_clusters, trainer, on_train_data=True, full=False):
        tp = [
            torch.zeros(class_clusters[i].clusters.shape[0]).to(self.device)
            for i in range(0, len(class_clusters))
        ]
        fp = [
            torch.zeros(class_clusters[i].clusters.shape[0]).to(self.device)
            for i in range(0, len(class_clusters))
        ]

        y_pred = []
        y_pred_prob = []
        y_true = []

        total_prob = 0
        
        if on_train_data:
            # dataloader = CacheLoader(trainer.train_loader, os.path.join(trainer.cfg.EXPERIMENT.OUT_DIR, trainer.cfg.EXPERIMENT.NAME, "cache_knn"), 
            dataloader = CacheLoader(trainer.train_loader, os.path.join(trainer.cfg.MODEL.BACKBONE_EXP_DIR, "cache_emb"), 
                                 "train_" + trainer.cfg.DATASET.TRAIN, self.model, self.device, self.EMB_SIZE)
        else:
            dataloader = CacheLoader(trainer.val_loader, os.path.join(trainer.cfg.MODEL.BACKBONE_EXP_DIR, "cache_emb"), 
            # dataloader = CacheLoader(trainer.val_loader, os.path.join(trainer.cfg.EXPERIMENT.OUT_DIR, trainer.cfg.EXPERIMENT.NAME, "cache_knn"), 
                                 "val_" + trainer.cfg.DATASET.VAL, self.model, self.device, self.EMB_SIZE)

        tbar = tqdm(range(0, len(dataloader)))
        # tbar = tqdm(trainer.train_loader) if on_train_data else tqdm(trainer.val_loader)
        for sample_id in tbar:
            # # [B, 3, H, W], [B, H, W]
            # image, target = sample[0].to(self.device), sample[1].to(self.device)
            # # [B * Hp * Wp, C]
            # emb = rearrange(self.model(image).emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> (b hp wp) c")
            sample = dataloader.get_id(sample_id)
            target, emb = sample["target"], sample["emb"]
            # [B * Hp * Wp, C]
            emb = rearrange(emb, "b hp wp c -> (b hp wp) c")

            # [B * Hp * Wp]
            target = self.preprocess_target(target)

            min_class_dist = torch.zeros(size=[emb.shape[0], len(class_clusters)]).to(self.device)
            min_class_dist_id = ( torch.zeros(size=[emb.shape[0], len(class_clusters)]).long().to(self.device))
            class_prob = torch.zeros(size=[emb.shape[0], len(class_clusters)]).to(self.device)
            for c in range(0, len(class_clusters)):
                cdists = self.knn_dist(emb, c, k_cluster=class_clusters[c])
                min_d = torch.min(cdists, dim=-1)
                min_class_dist[:, c] = min_d[0]
                min_class_dist_id[:, c] = min_d[1]
                class_prob[:, c] = torch.max((1.0 / (class_clusters[c].cluster_var[None, :].sqrt() * np.sqrt(2.0*np.pi))) * 
                                    torch.exp(-0.5 * cdists.pow(2) / class_clusters[c].cluster_var[None, :]), dim=-1)[0]

            y_pred.extend(torch.argmax(-min_class_dist, dim=-1).flatten().cpu().numpy())
            y_pred_prob.extend(torch.argmax(class_prob, dim=-1).flatten().cpu().numpy())
            y_true.extend(target.cpu().numpy())

            classification_id = torch.argmin(min_class_dist, dim=-1)
            valid_mask = target != self.IGNORE_LABEL
            correct_idx = torch.nonzero((target == classification_id) & valid_mask)
            incorrect_idx = torch.nonzero((target != classification_id) & valid_mask)
            for ci in correct_idx:
                class_id = target[ci]
                tp[class_id][min_class_dist_id[ci, class_id]] += 1

            for ci in incorrect_idx:
                class_id = classification_id[ci]
                fp[class_id][min_class_dist_id[ci, class_id]] += 1
            valid_idx = torch.nonzero(valid_mask)
            total_prob += torch.log(class_prob[valid_idx, target[valid_idx].long()]).sum().item()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_prob = np.array(y_pred_prob)
        cls_acc = []
        cls_acc_prob = []
        for c in np.unique(y_true):
            if c == self.IGNORE_LABEL:
                continue
            yc_true = y_true == c
            yc_pred = y_pred == c
            yc_pred_prob = y_pred_prob == c
            sum_c = float(np.sum(yc_true))
            cls_acc.append(np.sum(yc_true & yc_pred) / sum_c)
            cls_acc_prob.append(np.sum(yc_true & yc_pred_prob) / sum_c)
        P = [tp[i] / (tp[i] + fp[i]) for i in range(0, len(tp))]
        if full:
            return np.mean([torch.nanmean(P[i]).cpu().numpy() for i in range (0, len(tp))]), np.mean(cls_acc), np.mean(cls_acc_prob), total_prob
        else:
            return P

class GROODNetKNMSoftMultiClass(GROODNetKNM):
    def __init__(self, cfg):
        super(GROODNetKNMSoftMultiClass, self).__init__(cfg)
        self.NUM_KNN_TRIALS = 1 
        self.max_K = cfg.MODEL.MAX_K
        self.lr = cfg.OPTIMIZER.LR
        self.normalize_dist_by_tau = cfg.MODEL.TAU_NORM
        self.init_tau = cfg.MODEL.INIT_TAU
        self.pdf_dist_to_NP = cfg.MODEL.PDF_DIST2NP 
        self.knn_type = cfg.MODEL.KNN_TYPE
        self.reinit = cfg.MODEL.KNN_REINIT
        self.knn_subsample = cfg.MODEL.KNN_SUBSAMPLE

        # self._dist = self.knn_dist
        # def dist2sim_(dists, slope=0.5):
        #     return 1.0 / (1.0 + dists * slope)
        # def dist2sim_pdf_(dists, slope=0.5):
        #     # 100x to normalize values to roughly (0,1) range, so that the other GROOD assumptions hold
        #     return np.clip(100*dists, a_min=0, a_max=1)
        # self.dist2sim = dist2sim_ 
        # if self.pdf_dist_to_NP:
        #     self.dist2sim = dist2sim_pdf_

    def knn_dist(self, emb, class_id, k_cluster=None):
        if k_cluster is None:
            k_cluster = self.custom_data["knn_model"][class_id]
        diff = k_cluster._dist_fnc(
                emb,
                k_cluster.clusters.to(self.device),
                cov=k_cluster.cluster_var.to(self.device),
            )
        return diff

    # def compute_2dspace_vectors(self, out, eval_scale_factor=1):
    #     if eval_scale_factor > 1: 
    #         emb = rearrange( out.emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> b c hp wp")
    #         logits = rearrange(out.logits_embshape, "b hp wp c -> b c hp wp")
    #
    #         emb = torch.nn.functional.interpolate(emb, scale_factor=eval_scale_factor, mode="bilinear")
    #         logits = torch.nn.functional.interpolate(logits, scale_factor=eval_scale_factor, mode="bilinear")
    #
    #         emb = rearrange(emb, "b c hp wp -> (b hp wp) c")
    #         logits = rearrange(logits, "b c hp wp -> (b hp wp) c")
    #     else:
    #         # [B * Hp * Wp, C]
    #         emb = rearrange(out.emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> (b hp wp) c")
    #         logits = rearrange(out.logits_embshape, "b hp wp c -> (b hp wp) c")
    #
    #     self.custom_data["kmeans_mc_models"][0].to("cuda")
    #     nm_dist = self._dist(emb)
    #     self.custom_data["kmeans_mc_models"][0].to("cpu")
    #     return logits, nm_dist
    #
    # def knn_dist(self, emb):
    #     knn_model = self.custom_data["kmeans_mc_models"][0]
    #     with torch.no_grad():
    #         # [B, C, K]
    #         dist = -knn_model.sim_fnc(knn_model.norm_data(emb), tocpu=True)
    #         if self.pdf_dist_to_NP:
    #             dist = knn_model.eval_distribution_sum_large(dist, qnorm=True)
    #         else:
    #             dist = torch.min(dist, dim=-1)[0]
    #     return dist

    def fit_nm_model(self, trainer):
        max_epochs = 100
        lr = self.lr
        max_K = self.max_K
        max_init_data_per_class = 10000

        # dataloader = CacheLoader(trainer.train_loader, os.path.join(trainer.cfg.EXPERIMENT.OUT_DIR, trainer.cfg.EXPERIMENT.NAME, "cache_knn"), 
        #                         "train_" + trainer.cfg.DATASET.TRAIN, self.model, self.device, self.EMB_SIZE)
        dataloader = CacheLoader(trainer.train_loader, os.path.join(trainer.cfg.MODEL.BACKBONE_EXP_DIR, "cache_emb"), 
                                "train_" + trainer.cfg.DATASET.TRAIN, self.model, self.device, self.EMB_SIZE)

        print("Getting initialization data ...")
        # init data (take several batches of data to choose from)
        init_data = [torch.empty((0, self.EMB_SIZE))]*self.NUM_CLASSES
        # average batch size per class
        count_thr = [[] for _ in range(0, self.NUM_CLASSES)]
        dataloader.shuffle()
        for sample_id in tqdm(range(0, len(dataloader))):
            sample = dataloader.get_id(sample_id)
            target, emb = sample["target"], sample["emb"]
            # [B * Hp * Wp, C]
            emb = rearrange(emb, "b hp wp c -> (b hp wp) c")

            # [B * Hp * Wp]
            target = self.preprocess_target(target)

            for ul in torch.unique(target).cpu().numpy().tolist():
                if (ul == self.IGNORE_LABEL) or (init_data[ul].shape[0] > max_init_data_per_class):
                    continue
                mask = target == ul
                init_data[ul] = torch.cat([init_data[ul], emb[mask, :].cpu()], dim=0)
                count_thr[ul].append(mask.sum().item())
            
            if np.all([init_data[i].shape[0] > max_init_data_per_class for i in range(0, len(init_data))]):
                break

        count_thr = [int(np.mean(count_thr[i])) for i in range(0, len(count_thr))]
        print(f"Average batch sizes for classes: \n{*count_thr,}")

        init_labels = torch.cat([torch.ones(init_data[c].shape[0], dtype=torch.long)*c for c in range(0, self.NUM_CLASSES)], dim=0)
        init_data = torch.cat(init_data, dim=0)

        model = DiffKMeansMultiClassv2(self.NUM_CLASSES, max_K, self.EMB_SIZE,
                                       _data=init_data, _labels=init_labels,
                                       knn_type=self.knn_type, init_tau=self.init_tau).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

        def temperature_scheduler(it, max_iters, start_temp=1.0, end_temp=0.0):
            decay_ratio = it / max_iters
            coeff = 0.5 * (1.0 - np.cos(np.pi * decay_ratio)) # coeff ranges 0..1
            return start_temp + coeff * (end_temp - start_temp)

        print("Training K-NN model ...")
        loss_list = []
        iteration = 0
        for epoch in tqdm(range(0, max_epochs)):
            dataloader.shuffle()

            model.cluster_temp = torch.nn.Parameter(torch.tensor(temperature_scheduler(epoch, max_epochs, 0.5, 2), dtype=torch.float32), requires_grad=False)

            train_loss = 0.0
            train_loss_count = 0.0
            for sample_id in range(0, len(dataloader)):
                sample = dataloader.get_id(sample_id)
                target, emb = sample["target"], sample["emb"]
                # [B * Hp * Wp, C]
                emb = rearrange(emb, "b hp wp c -> (b hp wp) c")

                # [B * Hp * Wp]
                target = self.preprocess_target(target)
                
                optimizer.zero_grad()

                mask = target != self.IGNORE_LABEL
                if self.knn_subsample > 1:
                    # because of memory subsample datapoints
                    data_dict = SimpleNamespace(data = emb[mask, :][::self.knn_subsample, :].to(self.device), val_data=None, labels=target[mask][::self.knn_subsample].to(self.device), val_labels=None)
                else:
                    data_dict = SimpleNamespace(data = emb[mask, :].to(self.device), val_data=None, labels=target[mask].to(self.device), val_labels=None)
                loss = model.forward(data_dict)
                loss.backward()
                optimizer.step()               

                train_loss += loss.item()
                train_loss_count += 1
                iteration += 1

                if self.reinit and (epoch >= 1 and iteration % int(0.5 * len(dataloader)) == 0 and epoch < 0.9*max_epochs):
                    model.reset_clusters(data_dict)

            scheduler.step()

            loss_list.append(train_loss / train_loss_count)

        valid = []
        for c in range(0, model.num_classes):
            valid.append(model.mu_valid[c, :].sum().item())
        print (f"Valid per-class clusters (from total of {max_K}):\n {*valid,}")

        assignment_count = torch.zeros_like(model.mu_valid, dtype=torch.float32).to(self.device)
        for sample_id in range(0, len(dataloader)):
            sample = dataloader.get_id(sample_id)
            target, emb = sample["target"], sample["emb"]
            # [B * Hp * Wp, C]
            emb = rearrange(emb, "b hp wp c -> (b hp wp) c")
            # [B * Hp * Wp]
            target = self.preprocess_target(target)

            mask = target != self.IGNORE_LABEL
            _, r = model.assignment(model.norm_data(emb[mask, :].to(self.device)))

            for c in range(0, model.num_classes):
                data_mask = target[mask] == c
                valid_mu = model.mu_valid[c, :]
            
                assignment = torch.argmax(r[data_mask, c, :][:, valid_mu], dim=-1)
                one_hot = torch.nn.functional.one_hot(assignment, num_classes=valid_mu.sum().item()).float().sum(dim=0)
                assignment_count[c, valid_mu] += one_hot
                
        valid_assigned = assignment_count > 0
        model.running_assignment[~valid_assigned] = 0
        valid_a = []
        for c in range(0, model.num_classes):
            valid_a.append(valid[c] - valid_assigned[c, :].sum().item())
        print(f"Removed valid clusters with no assignment:\n {*valid_a,}")

        valid = []
        for c in range(0, model.num_classes):
            valid.append(model.mu_valid[c, :].sum().item())
        print (f"Final valid per-class clusters (from total of {max_K}):\n {*valid,}")

        model.to("cpu")
        if "kmeans_mc_models" in self.custom_data:
            self.custom_data["kmeans_mc_models"].append(model)
            self.custom_data["kmeans_mc_models_train_loss"].append(loss_list)
        else:
            self.custom_data["kmeans_mc_models"] = [model]
            self.custom_data["kmeans_mc_models_train_loss"] = [loss_list]

        # HACK: Copy final k-means clusters to the old data structure so the rest of the code can stay the same
        class_clusters = [
            OnlineKMeans(mahalanobis_dist=False, normalize_by_tau=self.normalize_dist_by_tau)
            for _ in range(0, self.NUM_CLASSES)
        ]
        for ul in range(0, self.NUM_CLASSES):
            class_clusters[ul].clusters = model.mu[ul, model.mu_valid[ul, :].detach()].detach().to(self.device)
            class_clusters[ul].cluster_var = model.tau[ul, model.mu_valid[ul, :].detach()].detach().to(self.device)
            # class_clusters[ul].cluster_var = torch.ones(class_clusters[ul].clusters.shape[0]).to(self.device) * (1 / model.cluster_temp)
            class_clusters[ul].cluster_weight = model.running_assignment[ul, model.mu_valid[ul, :]].detach().to(self.device)

        return class_clusters


# ======================================================================
# ======================================================================

class GROODNetKNMSoftMultiClassDeepLab(GROODNetKNMSoftMultiClass):
    def __init__(self, cfg):
        super(GROODNetKNMSoftMultiClassDeepLab, self).__init__(cfg)

    def compute_2dspace_vectors(self, out, eval_scale_factor=1):
        if eval_scale_factor > 1: 
            emb = rearrange( out.emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> b c hp wp")
            logits = rearrange(out.logits, "b hp wp c -> b c hp wp")

            emb = torch.nn.functional.interpolate(emb, scale_factor=eval_scale_factor, mode="bilinear")
            logits = torch.nn.functional.interpolate(logits, size=emb.shape[-2:], mode="bilinear")

            emb = rearrange(emb, "b c hp wp -> (b hp wp) c")
            logits = rearrange(logits, "b c hp wp -> (b hp wp) c")
        else:
            # [B * Hp * Wp, C]
            emb = rearrange(out.emb[:, :, :, -self.EMB_SIZE :], "b hp wp c -> (b hp wp) c")
            logits = rearrange(out.logits_embshape, "b hp wp c -> (b hp wp) c")

        nm_dist = torch.zeros_like(logits)
        for c in range(0, self.NUM_CLASSES):
            nm_dist[:, c] = self._dist(emb, class_id=c)
        return logits, nm_dist
