from unittest.mock import patch

import torch
import torchvision

from torchinfo import summary
from torch import nn
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
vit_model = torchvision.models.vit_b_16(weights=pretrained_weights).to(device)

for parameter in vit_model.parameters():
    parameter.requires_grad = False

classes = ["normal", "benign", "malignant"]

seed = 80
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)

# Setting up the model to process each patch.
# Starting off with 197x228 patches, with 1 channel as black and white image.
patch_width, patch_height = 197, 228
channels = 1
vit_model.heads = nn.Linear(in_features=patch_width * patch_height * channels, out_features=len(classes))


summ = summary(vit_model,
        input_size=(32, channels, patch_width, patch_height),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
print(summ)