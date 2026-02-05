import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(train_file, test_file, transform, batch_size):
    training_data = datasets.ImageFolder(train_file, transform=transform)
    test_data = datasets.ImageFolder(test_file, transform=transform)
    class_names = training_data.classes
    
    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=1,
                                 pin_memory=True)
    
    test_loader = DataLoader(test_data, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=1,
                             pin_memory=True)
    
    return training_loader, test_loader, class_names