import torch

def get_predictions(model, dataloader, device, class_names,
               print_stats=True):

    model.eval()

    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    correct = 0
    total = 0

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().view(-1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.view(-1).cpu())

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()


    accuracy = correct / total

    if print_stats:
        print("\n--- Test Results ---")
        print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        print("--- Per-Class Metrics ---")


    return all_probs, all_preds, all_labels

def test_model(model, dataloader, device, class_names, print_stats=True):
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
    stats = []
    if print_stats:
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
        stats.append([])
        stats[i].extend([class_name, TP, FP, FN, TN, precision, recall, f1, acc])

        if print_stats:
            print(f"\nClass: {class_name}")
            print(f"Confusion Table:")
            print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Accuracy:  {acc:.4f}")

    return stats


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
