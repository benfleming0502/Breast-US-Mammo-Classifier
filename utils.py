import torch
import torch.nn.functional as F
from PIL import Image
import os

def extract_patches(images, patch_size=224):
    B, C, H, W = images.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # symmetric padding
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom))

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    nH, nW = patches.shape[1], patches.shape[2]

    patches = patches.view(-1, C, patch_size, patch_size)

    return patches, (B, nH, nW)

def remove_bad_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(subdir, file)

            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue

            try:
                with Image.open(path) as img:
                    img.verify()  # quick check

                # reopen to fully load (verify alone isn't always enough)
                with Image.open(path) as img:
                    img.load()

            except Exception as e:
                print(f"Deleting corrupted file: {path}")
                try:
                    os.remove(path)
                except Exception as delete_error:
                    print(f"Failed to delete {path}: {delete_error}")