import torch
import torchvision

from torchinfo import summary
from torch import nn
from torchvision import transforms
from dataloading import get_dataloader
from training import train

import matplotlib.pyplot as plt
import torchvision
import CustomModels


def show_batch(dataloader, class_names, device, title="Sample Batch"):
    """
    Displays images from the dataloader's first batch to visualize the model's input
    """
    model_device = device
    images, labels = next(iter(dataloader))
    images = images.cpu()
    labels = labels.cpu()

    grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.title(title)
    plt.axis("off")

    # Printing the class labels of each image
    print("Labels:")
    for i in range(min(8, len(labels))):
        print(f"Image {i}: {class_names[labels[i]]}")

    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(f"Using {device} device")
    print(f"{torch.version.cuda}")

    patch_model = CustomModels.PatchTransformer()

    # for parameter in vit_model.parameters():
    #     parameter.requires_grad = False

    classes = ["normal", "benign", "malignant"]

    seed = 80
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    patch_model = CustomModels.PatchTransformer()


    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((2964, 2364)),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ) # Standard normalisation for use with ImageNet trained models
    ])

    training_images = "./training_mammograms"
    test_images = "./test_mammograms"

    training_loader, test_loader, class_names = get_dataloader(
        train_file=training_images,
        test_file=test_images,
        transform=image_transforms,
        batch_size=8
    )

    #show_batch(training_loader, class_names, device, title="Training Samples")
    #show_batch(test_loader, class_names, device, title="Test Samples")

    optimiser = torch.optim.Adam(params=patch_model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    vit_model_stats = train(
        patch_model=patch_model,
        training_dataloader=training_loader,
        optimiser=optimiser,
        loss_function=loss_function,
        epochs=15,
        device=device
    )

    print(vit_model_stats)

if __name__ == "__main__":
    main()
