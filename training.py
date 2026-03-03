import torch

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


def test_step(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            predictions = model(X)
            loss = loss_function(predictions, y)

            total_loss += loss.item() * X.size(0)
            preds = predictions.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(model,
          training_loader,
          test_loader,
          optimiser,
          loss_function,
          epochs,
          device):

    model.to(device)

    stats = {
        "training_loss": [],
        "training_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in range(epochs):
        training_loss, training_accuracy = train_step(
            model, training_loader, loss_function, optimiser, device
        )

        test_loss, test_accuracy = test_step(
            model, test_loader, loss_function, device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"training_loss: {training_loss:.4f} | "
            f"training_accuracy: {training_accuracy:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_accuracy: {test_accuracy:.4f}"
        )

        stats["training_loss"].append(training_loss)
        stats["training_accuracy"].append(training_accuracy)
        stats["test_loss"].append(test_loss)
        stats["test_accuracy"].append(test_accuracy)

    return stats
