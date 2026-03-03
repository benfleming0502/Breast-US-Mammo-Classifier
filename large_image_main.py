import torch
import torchvision

from torchinfo import summary
from torch import nn
from torchvision import transforms
from dataloading import get_dataloader
from training import train

def resize_vit(model, size):
    # Resizing the model to the new resolution, to replace the default 224x224
    patch_size = model.patch_embed.patch_size # Should be 16x16
    if size[0] % patch_size or size[1] % patch_size:
        print("image_size not compatible with patch_size")
        return None
    new_size = (size[0] // patch_size, size[1] // patch_size)

    embeddings = model.encoder.pos_embedding
    class_token = embeddings[:,:1]
    position = embeddings[:,1:]

    

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print(f"{torch.version.cuda}")


    vit_model = torchvision.models.vit_b_16().to(device)

    classes = ["normal", "benign", "malignant"]

    seed = 80
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    vit_model.heads = nn.Linear(in_features=64*48*3, out_features=len(classes))

    summ = summary(vit_model,
            input_size=(4, 3, 1024, 768),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

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