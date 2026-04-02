import torchvision.models as models
import torch.nn as nn
import torch
from utils import extract_patches

class PatchTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.frozen_backbone = True
        resnet_weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=resnet_weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

        self.cls_head = nn.Linear(d_model, 3)  # change for your task
    def unfreeze_backbone(self):
        self.frozen_backbone = False

    def forward(self, images):
        patches, (B, nH, nW) = extract_patches(images)

        if self.frozen_backbone:
            with torch.no_grad():
                feats = self.backbone(patches)
        else:
            feats = self.backbone(patches)
        # (B*nH*nW, 512, 7, 7)

        feats = feats.mean(dim=[2, 3])
        # (B*nH*nW, 512)

        feats = feats.view(B, nH * nW, 512)
        # (B, num_patches, 512)

        feats = self.transformer(feats)
        # (B, num_patches, 512)

        out = feats.mean(dim=1)
        # (B, 512)

        return self.cls_head(out)
