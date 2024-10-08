import os
import cv2
import torch
import numpy as np
import webp
from PIL import Image

from torch.utils.data import Dataset


class BRAVO(Dataset):
    def __init__(self, hparams):
        super().__init__()

        self.hparam = hparams
        self.split = hparams.dataset_mode
        all_splits = ['bravo_SMIYC', 'bravo_ACDC', 'bravo_outofcontext', 'bravo_synflare', 'bravo_synobjs', 'bravo_synrain']
        all_img_suffix = ['.jpg', '.png', '.png', '.png', '.png',  '.png']
        assert self.split in all_splits, f"split {self.split} not supported"
        split_idx = all_splits.index(self.split)
        self.img_suffix = all_img_suffix[split_idx]
        
        self.dataset_root = hparams.dataset_root
        self.images_root = os.path.join(self.dataset_root, self.split)
        self.images = []
        for (dirpath, dirnames, filenames) in os.walk(self.images_root):
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.images.append(dirpath + '/' + filename)
        
        self.images = [os.path.join(self.images_root, img_path) for img_path in self.images]
        self.labels = [''] * len(self.images)
        self.num_samples = len(self.images)

    def __getitem__(self, index):
        image = self.read_image(self.images[index])
        label = torch.zeros(image.shape[:2])
        
        filepath = self.images[index][len(self.dataset_root):]
        if filepath[0] == '/':
            filepath = filepath[1:]
        return image, label.type(torch.LongTensor), filepath

    def __len__(self):
        return self.num_samples

    @staticmethod
    def read_image(path):
        # img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = Image.open(path).convert('RGB')
        return torch.from_numpy(np.array(img))
