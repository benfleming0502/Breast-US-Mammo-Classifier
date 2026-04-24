import os
import torch
import torchvision

from torchinfo import summary
from torch import nn
from torchvision import transforms
from dataloading import get_dataloader
from training import train
from testing import test_model  # NEW
from LossFunctions import FocalLoss

import matplotlib.pyplot as plt
import CustomModels
import CustomTransforms

EPOCHS = 80
LEARNING_RATE = 5e-6

def train_model(patch_model, checkpoint, optimiser, device, total_epochs=EPOCHS, checkpoints_dir="./checkpoints", test_frequency=1, batch_size=4):
    seed = 80
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    image_transforms = transforms.Compose([
        transforms.Resize((2364, 2964)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        CustomTransforms.CLAHETransform(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    training_images = "./training_mammograms"
    test_images = "./test_mammograms"

    training_loader, test_loader, class_names = get_dataloader(
        train_file=test_images,
        test_file=test_images,
        transform=image_transforms,
        batch_size=batch_size
    )

    os.makedirs(checkpoints_dir, exist_ok=True)

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([176 / 76]).to(device)
    )

    vit_model_stats = train(
        patch_model=patch_model,
        training_dataloader=training_loader,
        optimiser=optimiser,
        loss_function=loss_function,
        epochs=total_epochs,
        device=device,
        checkpoint_dir=f"./{checkpoints_dir}",
        checkpoint=checkpoint,
        class_names=class_names,
        testing_dataloader=test_loader,
        test_freq=test_frequency
    )

def main(to_load=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(f"Using {device} device")
    print(f"{torch.version.cuda}")

    classes = ["normal", "abnormal"]

    patch_model = CustomModels.PatchTransformerResnet18().to(device)
    optimiser = torch.optim.Adam(params=patch_model.parameters(), lr=LEARNING_RATE)
    if to_load is None:
        epoch = 0
    else:
        checkpoint = torch.load(to_load)
        patch_model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    train_model(
        patch_model=patch_model,
        checkpoint=epoch,
        optimiser=optimiser,
        device=device
    )





if __name__ == "__main__":
    main()
