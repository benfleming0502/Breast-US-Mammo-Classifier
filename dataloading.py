import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(train_file, test_file, transform, batch_size):
    training_data = datasets.ImageFolder(train_file, transform=transform)
    test_data = datasets.ImageFolder(test_file, transform=transform)
    class_names = training_data.classes

    # for i in range(len(training_data)):
    #     print(i, training_data.imgs[i][0])
    #     image = training_data[i]
    #     if image[0].size() != torch.Size([3, 2964, 2364]):
    #         print(image[0].size())

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=8,
                                 persistent_workers=True,
                                 prefetch_factor=2)#,
                                 #pin_memory=True)

    test_loader = DataLoader(test_data, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=8,
                             persistent_workers=True,
                             prefetch_factor=2,
                             pin_memory=True)
    
    return training_loader, test_loader, class_names
#
# image_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((2364, 2964)),
#     transforms.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225)
#     ) # Standard normalisation for use with ImageNet trained models
# ])
#
# training_images = "./training_mammograms"
# test_images = "./test_mammograms"
#
# training_loader, test_loader, class_names = get_dataloader(
#     train_file=training_images,
#     test_file=test_images,
#     transform=image_transforms,
#     batch_size=16
# )