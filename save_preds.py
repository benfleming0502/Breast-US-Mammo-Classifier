# Tests all checkpoints saved by the given model
# Outputs the data as a csv
import csv

import numpy as np
from torchvision import transforms
import torch
from dataloading import get_test_dataloader
import os
from CustomModels import *
from torchvision import models
import testing
import pandas as pd
import CustomTransforms


def main():
    save_path = "./"
    best_t18 = "./patch18-checkpoints-lowres/checkpoint_epoch_53.pth"
    best_r18 = "./resnet18-checkpoints-lowres/checkpoint_epoch_41.pth"
    best_t50 = "./patch50-checkpoints-lowres/checkpoint_epoch_64.pth"
    best_r50 = "./resnet50-checkpoints-lowres/checkpoint_epoch_80.pth"

    test_images_dir = "./test_mammograms"

    image_transforms = transforms.Compose([
            transforms.Resize((591, 741)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    test_loader, class_names = get_test_dataloader(
            test_file=test_images_dir,
            transform=image_transforms,
            batch_size=1
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_t18 = PatchTransformerResnet18()

    optimiser = torch.optim.Adam(model_t18.parameters())
    checkpoint = torch.load(best_t18)
    model_t18.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    model_t18.to(device)

    probs_t18, preds_t18, labels = testing.get_predictions(model_t18, test_loader, device, class_names=class_names)

    del model_t18
    del optimiser
    torch.cuda.empty_cache()

    model_r18 = PatchResnet18()

    optimiser = torch.optim.Adam(model_r18.parameters())
    checkpoint = torch.load(best_r18)
    model_r18.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    model_r18.to(device)

    probs_r18, preds_r18, labels = testing.get_predictions(model_r18, test_loader, device, class_names=class_names)

    del model_r18
    del optimiser
    torch.cuda.empty_cache()

    model_t50 = PatchTransformerResnet50()

    optimiser = torch.optim.Adam(model_t50.parameters())
    checkpoint = torch.load(best_t50)
    model_t50.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    model_t50.to(device)

    probs_t50, preds_t50, labels = testing.get_predictions(model_t50, test_loader, device, class_names=class_names)

    del model_t50
    del optimiser
    torch.cuda.empty_cache()

    model_r50 = PatchResnet50()

    optimiser = torch.optim.Adam(model_r50.parameters())
    checkpoint = torch.load(best_r50)
    model_r50.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    model_r50.to(device)

    probs_r50, preds_r50, labels = testing.get_predictions(model_r50, test_loader, device, class_names=class_names)

    del model_r50
    del optimiser
    torch.cuda.empty_cache()

    # Make all into one csv file with labels and the probabilities and predictions for each model
    print("Shapes:")
    print("t18 preds:", np.array(preds_t18).shape)
    print("t18 probs:", np.array(probs_t18).shape)
    print("labels:", np.array(labels).shape)

    probs_t18 = np.array(probs_t18).reshape(-1)
    preds_t18 = np.array(preds_t18).reshape(-1)

    probs_r18 = np.array(probs_r18).reshape(-1)
    preds_r18 = np.array(preds_r18).reshape(-1)

    probs_t50 = np.array(probs_t50).reshape(-1)
    preds_t50 = np.array(preds_t50).reshape(-1)

    probs_r50 = np.array(probs_r50).reshape(-1)
    preds_r50 = np.array(preds_r50).reshape(-1)

    labels = np.array(labels).reshape(-1)

    assert len(preds_t18) == len(labels)
    assert len(preds_r18) == len(labels)
    assert len(preds_t50) == len(labels)
    assert len(preds_r50) == len(labels)

    df = pd.DataFrame({
        "label": labels,
        "t18_pred": preds_t18,
        "r18_pred": preds_r18,
        "t50_pred": preds_t50,
        "r50_pred": preds_r50,
        "t18_prob": probs_t18,
        "r18_prob": probs_r18,
        "t50_prob": probs_t50,
        "r50_prob": probs_r50
    })

    csv_path = os.path.join(save_path, "model_predictions_lowres.csv")
    df.to_csv(csv_path, index=False)



if __name__ == "__main__":
    main()