import os
import sys
import torch
import importlib
import numpy as np
from PIL import Image
from types import SimpleNamespace
from einops import rearrange


def get_experiment_cfg(config, code_dir, load_git_code=False):
    #from config import get_cfg_defaults
    config_module = importlib.util.spec_from_file_location("get_cfg_defaults", os.path.join(code_dir, "config", "defaults.py")).loader.load_module()
    cfg_fnc = getattr(config_module, "get_cfg_defaults")
    cfg_local = cfg_fnc()
    # read the experiment parameters
    if os.path.isfile(config):
        with open(config, 'r') as f:
            cc = cfg_local._load_cfg_from_yaml_str(f)
        cfg_local.merge_from_file(config)
        cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME
    else:
        raise RuntimeError(f"Config file does not exist: {config}")
    return cfg_local

class PixOOD():
    def __init__(self, config, code_dir, **kwargs_global) -> None:
        self.code_dir = code_dir
        # store all loaded modules so it can be later restored to the same state
        pre_modules_keys = []
        for k, _ in sys.modules.items():
            pre_modules_keys.append(k)

        cfg_local = get_experiment_cfg(config, self.code_dir )
        cfg_local.EXPERIMENT.OUT_DIR = os.path.abspath("./")

        checkpoint_name = os.path.join("./", "chkpts", os.path.basename(config)[:-4] +  "pth")
        if os.path.isfile(checkpoint_name):
            cfg_local.EXPERIMENT.RESUME_CHECKPOINT = checkpoint_name
        else:
            raise RuntimeError(f"\033[101m=> no checkpoint found at:\033[0m {checkpoint_name}")

        # CUDA
        if not torch.cuda.is_available():
            print ("GPU is disabled")
            cfg_local.SYSTEM.USE_GPU = False

        self.cfg = cfg_local
        self.device = torch.device("cuda" if cfg_local.SYSTEM.USE_GPU else "cpu")

        sys.path.insert(0, self.code_dir)
        kwargs = {'cfg': cfg_local}
        spec = importlib.util.spec_from_file_location(cfg_local.MODEL.FILENAME, os.path.join(self.code_dir, "net", "models", cfg_local.MODEL.FILENAME + ".py"))
        model_module = spec.loader.load_module()
        print (self.code_dir, model_module)
        self.model = getattr(model_module, cfg_local.MODEL.NET)(**kwargs)
        # load input preprocessing for the network
        spec = importlib.util.spec_from_file_location("augmentations", os.path.join(self.code_dir, "dataloaders", "augmentations.py"))
        augment_module = spec.loader.load_module()
        self.transforms = getattr(augment_module, cfg_local.DATASET.AUGMENT)().test(cfg_local)
        # clean up the inserted code path
        sys.path = sys.path[1:]

        # load the model paraters
        if cfg_local.EXPERIMENT.RESUME_CHECKPOINT is not None:
            checkpoint = torch.load(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")
            for key in list(checkpoint['state_dict'].keys()):
                if '_orig_mod.' in key:
                    checkpoint['state_dict'][key.replace('_orig_mod.', '')] = checkpoint['state_dict'][key]
                    del checkpoint['state_dict'][key]
            
            strict = not checkpoint.get("save_trainable_only", False)
            if not strict:
                print ("Saved model stores only tranable weights of model --> disabling strict model loading")
                model_state = self.model.state_dict()
                no_match = { k:v.size() for k,v in checkpoint['state_dict'].items() if (k in model_state and v.size() != model_state[k].size()) or (k not in model_state)}
                print("Number of not matched parts: ", len(no_match))
                print("-----------------")
                print(no_match)
                print("-----------------")

            self.model.load_state_dict(checkpoint['state_dict'], strict=strict)
            custom_data = checkpoint.get("custom_data", {})
            if hasattr(self.model, "custom_data"):
                self.model.custom_data = custom_data
            
            print("\033[92m=> loaded checkpoint '{}' (epoch {})\033[0m".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, checkpoint['epoch']))
            del checkpoint

        # Using cuda
        self.model.to(self.device)
        self.model.eval()

        # clean-up imported modules
        to_del = []
        for k, _ in sys.modules.items():
            if k not in pre_modules_keys and any(m in k for m in ["config", "dataloaders", "helpers", "net"]):
                to_del.append(k)
        for k in to_del:
            del sys.modules[k]

        # Cityscapes labels
        # 0:road 1:sidewalk 2:building 3:wall 4:fence
        # 5:pole 6:traffic light 7:traffic sign
        # 8:vegetation 9:terrain 10:sky 11:person
        # 12:rider 13:car 14:truck 15:bus
        # 16:train 17:motorcycle 18:bicycle
        if "eval_labels" not in kwargs_global.keys():
            self.eval_labels = [0, 1] 
            print(f"Using default road+sidewalk labels for anomaly detection!")
        elif len(kwargs_global["eval_labels"]) == 0:
            self.eval_labels = np.arange(self.cfg.MODEL.NUM_CLASSES).tolist()
            print(f"Using all labels for anomaly detection: {*self.eval_labels,}")
        else:
            self.eval_labels =  kwargs_global["eval_labels"]
            print(f"Using labels {*self.eval_labels,} for anomaly detection!")

        self.eval_scale_factor = kwargs_global.get("eval_scale_factor", 1)
        print(f"Using emb scale factor {self.eval_scale_factor}")

        if hasattr(self.model, "inference_flag"):
            self.model.inference_flag = True

        self.norm_scale = kwargs_global.get("norm_scale", 0.2)
        self.norm_thr = kwargs_global.get("norm_thr", 0.95)

        # EAFP approach
        try:
            print("Running before_eval fnc. of the model to set up dynamic structures.")
            self.model.before_eval()
        except AttributeError:
            print("   - Model does not have before_eval fnc.")

        self.colors = np.array([[128, 64, 128],                     # 0: road
                                [244, 35, 232],                     # 1: sidewalk
                                [70, 70, 70],                       # 2: building
                                [102, 102, 156],                    # 3: wall
                                [190, 153, 153],                    # 4: fence
                                [153, 153, 153],                    # 5: pole
                                [250, 170, 30],                     # 6: traffic_light
                                [220, 220, 0],                      # 7: traffic_sign
                                [107, 142, 35],                     # 8: vegetation
                                [152, 251, 152],                    # 9: terrain
                                [0, 130, 180],                      # 10: sky
                                [220, 20, 60],                      # 11: person
                                [255, 0, 0],                        # 12: rider
                                [0, 0, 142],                        # 13: car
                                [0, 0, 70],                         # 14: truck
                                [0, 60, 100],                       # 15: bus
                                [0, 80, 100],                       # 16: train
                                [0, 0, 230],                        # 17: motorcycle
                                [119, 11, 32],                      # 18: bicycle
                                [0, 0, 0]])                         # 19: unlabelled


    def evaluate(self, input, return_anomaly_score=False):
        orig_size = input.shape[:2]
        #assumes single pil image
        input_sn = self.transforms(SimpleNamespace(image=Image.fromarray(input.numpy()), label=None, image_name=""))
        x = input_sn.image.to(self.device)[None, ...]
        with torch.no_grad():
            out = self.model(x, eval_scale_factor=self.eval_scale_factor)

        # convert outputs to the original pil image resolution 
        # "b h w" 
        pred_score_hires = torch.nn.functional.interpolate(out.pred_score[:, None, ...], size=orig_size, mode="nearest").squeeze().cpu()
        pred_score_hires_all = torch.nn.functional.interpolate(rearrange(out.pred_score_all, "b h w c -> b c h w"), size=orig_size, mode="bilinear")
        pred_score_hires_all = rearrange(pred_score_hires_all, "b c h w -> b h w c").squeeze().cpu()
        pred_y_hires = torch.nn.functional.interpolate(out.pred_y.float()[:, None, ...], size=orig_size, mode="nearest").long()
        pred_y_hires = pred_y_hires.squeeze().cpu()

        s = (1.0- pred_score_hires)
        mask_id = s <= self.norm_thr 
        mask_ood = s > self.norm_thr 
        s[mask_id] = self.norm_scale * (s[mask_id]/self.norm_thr)
        s[mask_ood] = self.norm_scale + (1.0 - self.norm_scale)*((s[mask_ood]-self.norm_thr) / (1.0 - self.norm_thr))

        seg_pred_img = Image.fromarray(pred_y_hires.byte().cpu().numpy())
        seg_pred_img.putpalette(self.colors.astype("uint8"))
        seg_pred_img = Image.blend(Image.fromarray(input.numpy()), seg_pred_img.convert("RGB"), alpha=0.5)

        return SimpleNamespace(pred_y=pred_y_hires,
                               anomaly_score=s,
                               out_segm_img = seg_pred_img)


