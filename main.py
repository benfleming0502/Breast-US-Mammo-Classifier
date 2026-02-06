import torch
import torchvision

from torchinfo import summary
from torch import nn
from torchvision import transforms
from dataloading import get_dataloader
from training import train

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print(f"{torch.version.cuda}")


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

    # patch_width, patch_height = 197, 228
    # channels = 1
    # vit_model.heads = nn.Linear(in_features=patch_width * patch_height * channels, out_features=len(classes))


    # summ = summary(vit_model,
    #         input_size=(32, channels, patch_width, patch_height),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])
    # print(summ)

    vit_model.heads = nn.Linear(in_features=16**2 * 3, out_features=len(classes))

    summ = summary(vit_model,
            input_size=(4, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])


    pretrained_transforms = pretrained_weights.transforms()
    print(pretrained_transforms)

    training_images = "./training_ultrasounds"
    test_images = "./test_ultrasounds"
    training_loader, test_loader, class_names = get_dataloader(train_file=training_images, test_file=test_images, transform=pretrained_transforms, batch_size=4)

    optimiser = torch.optim.Adam(params=vit_model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    vit_model_stats = train(model=vit_model,
                            training_loader=training_loader,
                            test_loader=test_loader,
                            optimiser=optimiser,
                            loss_function=loss_function,
                            epochs=5,
                            device=device)
    print(vit_model_stats)

if __name__ == "__main__":
    main()