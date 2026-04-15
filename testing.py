import torch

def test_model(model, dataloader, device, class_names):
    model.eval()

    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().view(-1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print("\n--- Test Results ---")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})\n")

    # Per-class metrics
    print("--- Per-Class Metrics ---")
    for i, class_name in enumerate(class_names):
        TP = confusion_matrix[i, i].item()
        FP = confusion_matrix[:, i].sum().item() - TP
        FN = confusion_matrix[i, :].sum().item() - TP
        TN = confusion_matrix.sum().item() - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = (TP + TN) / confusion_matrix.sum().item()

        print(f"\nClass: {class_name}")
        print(f"Confusion Table:")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {acc:.4f}")

    return accuracy, confusion_matrix


def test_model_accuracy(model, dataloader, device, class_names):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print("\n--- Test Results ---")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy
