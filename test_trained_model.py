import torch
from CustomModels import PatchTransformerResnet18
from dataloading import get_dataloader
from torchvision import transforms
from testing import test_model

def main():
    model = PatchTransformerResnet18()
    optimiser = torch.optim.Adam(model.parameters())

    checkpoint = torch.load("checkpoints/checkpoint_epoch_50.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

    training_images = "./training_mammograms"
    test_images = "./test_mammograms"

    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((2364, 2964)),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20)
    ])

    training_loader, test_loader, class_names = get_dataloader(
            train_file=training_images,
            test_file=test_images,
            transform=image_transforms,
            batch_size=4
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    print(test_model(model, test_loader, device, class_names))

if __name__ == "__main__":
    main()