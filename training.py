import torch

from tqdm.auto import tqdm

def train_step(model, dataloader, loss_function, optimiser, device):
    model.train()
    loss, accuracy = 0,0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        predictions = model(X)
        batch_loss = loss_function(predictions, y)
        loss += batch_loss.item() 

        optimiser.zero_grad()
        batch_loss.backward()
        optimiser.step()

        y_pred_class = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
        accuracy += (y_pred_class == y).sum().item()/len(predictions)

    loss = loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    return loss, accuracy

def test_step(model, dataloader, loss_function, device):
    model.eval() 
    loss, accuracy = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            predictions = model(X)

            batch_loss = loss_function(predictions, y)
            loss += batch_loss.item()

            labels = predictions.argmax(dim=1)
            accuracy += ((labels == y).sum().item()/len(labels))

    loss = loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    return loss, accuracy

def train(model, 
          training_loader, 
          test_loader, 
          optimiser, 
          loss_function, 
          epochs, 
          device):
    model.to(device)
    stats = {"training_loss": [],
               "training_accuracy": [],
               "test_loss": [],
               "test_accuracy": []
    }

    for epoch in tqdm(range(epochs)):
        training_loss, training_accuracy = train_step(model=model,
                                          dataloader=training_loader,
                                          loss_function=loss_function,
                                          optimiser=optimiser,
                                          device=device)
        
        test_loss, test_accuracy = test_step(model=model,
          dataloader=test_loader,
          loss_function=loss_function,
          device=device)

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