# Training script used for patched images
# The training images are split into as many 224x224 blocks as possible, creating extra blackspace where edge is found
# Then these blocks are ran through a resnet-18 feature extractor (No classifier head) to bring each down to a 7x7 block
# Then these blocks are flattened and concatenated with the rest and run through the given transformer model.

import torch
from sympy.stats.rv import probability

from testing import test_model
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
          freeze_point=4,
          checkpoint_dir=None,
          testing_dataloader=None,
          class_names = None,
          checkpoint=0,
          checkpoint_freq=1,
          test_freq=1):
    batch_size = training_dataloader.batch_size
    acc_cons = []
    best_acc = 0
    best_acc_i = 0
    patch_model.to(device)
    for epoch in range(checkpoint, epochs):
        i = 0
        patch_model.train()
        if epoch == freeze_point:
            patch_model.unfreeze_backbone()
        total_loss = 0
        total_correct = 0
        for images, labels in training_dataloader:
            i += 1
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = patch_model(images)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct = (preds == labels).sum()
            total_correct += correct

            loss = loss_function(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if i == -1: # Change to number ~ 0-150 for a random sample of a batch's prediction
                probs = torch.sigmoid(outputs[:10])
                preds = (probs > 0.5).float()

                print("probs:", probs)
                print("preds:", preds)
                print("labels:", labels[:10])
                print("loss:", loss.item())
            total_loss += loss.item()
        training_accuracy = total_correct / (i+1)
        end_loss_avg = total_loss / ((i + 1) * batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {end_loss_avg:.4f}")
        print(f"Training Accuracy: {training_accuracy:.4f}")

        # ✅ Save checkpoint every 2 epochs
        if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': patch_model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': end_loss_avg,
                'training_accuracy': training_accuracy
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")
        if (not (testing_dataloader is None or class_names is None)) and test_freq != 0 and (epoch + 1) % test_freq == 0:
            print(f"Testing Model at epoch {epoch + 1}:\n")
            cuurent_stats = test_model(patch_model, testing_dataloader, device, class_names)
            current_acc = cuurent_stats[0][-1]
            acc_cons.append([current_acc, cuurent_stats])
            if current_acc > best_acc:
                best_acc = current_acc
                best_acc_i = epoch // test_freq
    if test_freq != 0:
        print(f"Best epoch was: {best_acc_i + 1}")
        print(f"With accuracy: {acc_cons[best_acc_i][0]}")
        print(f"Stats:\n{acc_cons[best_acc_i][1]}")
    checkpoint_path = f"{checkpoint_dir}/final_model.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': patch_model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss.item()
    }, checkpoint_path)

    print("Final model saved.")
