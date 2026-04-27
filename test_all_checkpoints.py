# Tests all checkpoints saved by the given model
# Outputs the data as a csv
import csv

from torchvision import transforms
import torch
from dataloading import get_test_dataloader
import os
from CustomModels import *
from torchvision import models
import testing


def test_checkpoint(model, test_loader, checkpoint_dir, device, class_names):

    optimiser = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)

    accuracy, confusion_matrix = testing.test_model(model, test_loader, device, class_names, print_stats=False)

    return accuracy, confusion_matrix

def test_models_checkpoints(model, test_images_dir, checkpoints_dir, device, model_name, image_transforms=None):
    if image_transforms is None:
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((2364, 2964)),
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

    test_data = []
    headers = ["Epoch", "Class", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Accuracy"]
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            checkpoint_num = file.split("_")[-1][:-4]
            print(f"Epoch {checkpoint_num} Testing...\n")
            checkpoint_data = test_checkpoint(model, test_loader, root + "/" + file , device, class_names)
            print(headers)
            for j in range(len(checkpoint_data)):
                observation = [checkpoint_num]
                observation.extend(checkpoint_data[j])
                test_data.append(observation)
                print(observation)



    print(test_data)

    with open(model_name + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(test_data)
    f.close()



def main():
    model = PatchTransformerResnet50()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./patch50-preproc-checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "patch-50-preproc")

if __name__ == "__main__":
    main()