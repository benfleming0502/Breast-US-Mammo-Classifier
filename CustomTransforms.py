import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        clahe_img = clahe.apply(gray)

        clahe_img = clahe_img.astype(np.float32) / 255.0
        clahe_img = np.stack([clahe_img] * 3, axis=-1)

        clahe_img = torch.from_numpy(clahe_img).permute(2, 0, 1)

        return clahe_img
