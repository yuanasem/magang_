import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from model import TBXRayModel
from visualizations import generate_comprehensive_report
import os
from PIL import Image

def create_dataloaders(data_dir, batch_size, train_transform, val_transform):
    # Define class names
    class_names = ['Healthy', 'Sick', 'TB']

    # Load dataset using ImageFolder from TBX11K/imgs/ (Healthy, Sick, TB)
    try:
        full_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=None,
            loader=lambda path: Image.open(path).convert('L'),  # Load as grayscale
            is_valid_file=lambda path: any(cls in path for cls in class_names)  # Filter required classes
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {data_dir}: {e}")

    # Verify class names
    detected_classes = full_dataset.classes
    if not all(cls in detected_classes for cls in class_names):
        raise ValueError(f"Expected classes {class_names}, but found {detected_classes}")

    # Filter samples to include only Healthy, Sick, TB
    valid_samples = [
        (path, target) for path, target in full_dataset.samples
        if full_dataset.classes[target] in class_names
    ]
    full_dataset.samples = valid_samples
    full_dataset.classes = class_names
    full_dataset.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    # Get labels and indices
    labels = [target for _, target in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    # Validate labels
    unique_labels = np.unique(labels)
    expected_labels = np.arange(len(class_names))
    if not np.all(np.isin(unique_labels, expected_labels)):
        raise ValueError(f"Unexpected label values {unique_labels}, expected {expected_labels}")

    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    # Create subsets for train and validation
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load test dataset from TBX11K/imgs/test/
    try:
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'test'),
            transform=val_transform,
            loader=lambda path: Image.open(path).convert('L'),  # Load as grayscale
            is_valid_file=lambda path: any(cls in path for cls in class_names)
        )
        # Filter test dataset to include only Healthy, Sick, TB
        test_valid_samples = [
            (path, target) for path, target in test_dataset.samples
            if test_dataset.classes[target] in class_names
        ]
        test_dataset.samples = test_valid_samples
        test_dataset.classes = class_names
        test_dataset.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    except Exception as e:
        print(f"Warning: Failed to load test dataset from {os.path.join(data_dir, 'test')}: {e}")
        test_dataset = None
        test_loader = None
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    # Validate class distribution
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    train_class_counts = np.bincount(train_labels, minlength=len(class_names))
    val_class_counts = np.bincount(val_labels, minlength=len(class_names))

    print(f"Training class distribution: {dict(zip(class_names, train_class_counts))}")
    print(f"Validation class distribution: {dict(zip(class_names, val_class_counts))}")

    if test_dataset is not None:
        test_labels = [target for _, target in test_dataset.samples]
        test_class_counts = np.bincount(test_labels, minlength=len(class_names))
        print(f"Test class distribution: {dict(zip(class_names, test_class_counts))}")
    else:
        print("Test dataset not loaded; skipping test distribution.")

    if np.any(val_class_counts == 0) or np.any(train_class_counts == 0):
        raise ValueError("Some classes have no samples in train or validation set. Check data.")

    return train_loader, val_loader, test_loader, class_names

def train_model(
    data_dir,
    model_output_path="best_model_updated.pth",
    num_epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    patience=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms for single-channel grayscale
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Single-channel normalization
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Single-channel normalization
    ])

    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir, batch_size, train_transforms, val_transforms
    )

    writer = SummaryWriter()
    try:
        model = TBXRayModel(num_classes=3, pretrained=True).to(device)
        model.convnext.set_grad_checkpointing(True)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

    # Class weights
    train_labels = [target for _, target in train_loader.dataset.dataset.samples]
    train_labels = [train_labels[i] for i in train_loader.dataset.indices]
    
    # Validate train_labels
    unique_train_labels = np.unique(train_labels)
    expected_labels = np.arange(len(class_names))
    if not np.all(np.isin(unique_train_labels, expected_labels)):
        raise ValueError(f"Unexpected train label values {unique_train_labels}, expected {expected_labels}")

    class_sample_count = np.bincount(train_labels, minlength=len(class_names))
    print(f"Class sample counts: {dict(zip(class_names, class_sample_count))}")
    class_sample_count[class_sample_count == 0] = 1  # Avoid division by zero
    weight = 1. / class_sample_count
    class_weights = torch.FloatTensor(weight).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_labels, best_predictions, best_probabilities = [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs, labels = inputs.to(device), labels.to(device)

            if torch.any(torch.isnan(inputs)) or torch.any(torch.isinf(inputs)):
                print(f"Warning: NaN/Inf in input data at batch {batch_idx}. Skipping.")
                continue

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                    print(f"Warning: NaN/Inf in outputs at batch {batch_idx}. Skipping.")
                    continue
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader) if train_total > 0 else float('inf')
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_labels, all_predictions, all_probabilities = [], [], []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validation")):
                inputs, labels = inputs.to(device), labels.to(device)

                if torch.any(torch.isnan(inputs)) or torch.any(torch.isinf(inputs)):
                    print(f"Warning: NaN/Inf in validation input at batch {batch_idx}. Skipping.")
                    continue

                with autocast():
                    outputs = model(inputs)
                    if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                        print(f"Warning: NaN/Inf in validation outputs at batch {batch_idx}. Skipping.")
                        continue
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                    print(f"Warning: NaN/Inf in probabilities at batch {batch_idx}. Skipping.")
                    continue
                _, predicted = torch.max(probs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader) if val_total > 0 else float('inf')
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_f1 = f1_score(all_labels, all_predictions, average='weighted') if all_labels else 0.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)

        print(f"\nEpoch {epoch+1} Metrics:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Per-class accuracy
        class_acc = {}
        for i in range(len(class_names)):
            mask = np.array(all_labels) == i
            if mask.sum() > 0:
                class_acc[class_names[i]] = accuracy_score(
                    np.array(all_labels)[mask], np.array(all_predictions)[mask]
                )
            else:
                class_acc[class_names[i]] = float('nan')
                print(f"Warning: No samples for {class_names[i]} in this epoch")
        for cls, acc in class_acc.items():
            print(f"{cls} Accuracy: {acc:.4f}" if not np.isnan(acc) else f"{cls} Accuracy: N/A")

        # Save best model and predictions
        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), model_output_path)
                print(f"Saved best model to {model_output_path}")
            except Exception as e:
                print(f"Warning: Failed to save model to {model_output_path}: {e}")
            best_labels = all_labels
            best_predictions = all_predictions
            best_probabilities = all_probabilities
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Test set evaluation (if test_loader exists)
    if test_loader is not None:
        print("\nEvaluating on test set...")
        try:
            model.load_state_dict(torch.load(model_output_path, map_location=device))
            model.eval()
        except Exception as e:
            print(f"Warning: Failed to load model for testing: {e}")
            return model, train_losses, val_losses, train_accuracies, val_accuracies

        test_loss, test_correct, test_total = 0, 0, 0
        test_labels, test_predictions, test_probabilities = [], [], []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
                inputs, labels = inputs.to(device), labels.to(device)

                if torch.any(torch.isnan(inputs)) or torch.any(torch.isinf(inputs)):
                    print(f"Warning: NaN/Inf in test input at batch {batch_idx}. Skipping.")
                    continue

                with autocast():
                    outputs = model(inputs)
                    if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                        print(f"Warning: NaN/Inf in test outputs at batch {batch_idx}. Skipping.")
                        continue
                    loss = criterion(outputs, labels)

                test_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                    print(f"Warning: NaN/Inf in test probabilities at batch {batch_idx}. Skipping.")
                    continue
                _, predicted = torch.max(probs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_labels.extend(labels.cpu().numpy())
                test_predictions.extend(predicted.cpu().numpy())
                test_probabilities.extend(probs.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader) if test_total > 0 else float('inf')
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        test_f1 = f1_score(test_labels, test_predictions, average='weighted') if test_labels else 0.0

        print(f"\nTest Metrics:")
        print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | Test F1: {test_f1:.4f}")

        # Per-class test accuracy
        test_class_acc = {}
        for i in range(len(class_names)):
            mask = np.array(test_labels) == i
            if mask.sum() > 0:
                test_class_acc[class_names[i]] = accuracy_score(
                    np.array(test_labels)[mask], np.array(test_predictions)[mask]
                )
            else:
                test_class_acc[class_names[i]] = float('nan')
                print(f"Warning: No test samples for {class_names[i]}")
        for cls, acc in test_class_acc.items():
            print(f"{cls} Test Accuracy: {acc:.4f}" if not np.isnan(acc) else f"{cls} Test Accuracy: N/A")

        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_scalar('F1/test', test_f1, epoch)

    # Generate report for validation set
    print("\nGenerating comprehensive evaluation report...")
    if best_labels and best_predictions and not np.any(np.isnan(best_probabilities)):
        try:
            generate_comprehensive_report(
                best_labels,
                best_predictions,
                best_probabilities,
                class_names,
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                save_dir="./results"
            )
        except Exception as e:
            print(f"Warning: Failed to generate report: {e}")
    else:
        print("Skipping report generation due to invalid data.")

    writer.close()
    print("Training and evaluation complete.")
    return model, train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == '__main__':
    DATA_DIR = 'TBX11K/imgs'  # Root directory containing Healthy, Sick, TB, test
    train_model(DATA_DIR)
