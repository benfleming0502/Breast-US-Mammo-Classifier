# Training script used for patched images
# The training images are split into as many 224x224 blocks as possible, creating extra blackspace where edge is found
# Then these blocks are ran through a resnet-18 feature extractor (No classifier head) to bring each down to a 7x7 block
# Then these blocks are flattened and concatenated with the rest and run through the given transformer model.

import torch
import torchvision.models as models
import torch.nn as nn
import torchinfo

def train_step(model, dataloader, loss_function, optimiser, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        predictions = model(X)
        loss = loss_function(predictions, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * X.size(0)
        preds = predictions.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += X.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(patch_model,
          training_dataloader,
          optimiser,
          loss_function,
          epochs,
          device,
          freeze_point=10):

    patch_model.to(device)
    for epoch in range(epochs):
        if epoch == freeze_point:
            try:
                patch_model.unfreeze_backbone()
            except:
                print(f"Epoch {epoch}: Freeze backbone failed")

        patch_model.train()
        for images, labels in training_dataloader:
            images = images.to(device)
            # labels = labels.to(device).float().unsqueeze(1)
            labels = labels.to(device).long()
            optimiser.zero_grad()

            outputs = patch_model(images)
            # print(outputs, labels)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch}: {loss.item():.4f}")