
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os

def save_training_plots(train_losses, val_losses, train_accuracies=None, val_accuracies=None, save_dir="./results"):
    """Save training and validation loss/accuracy plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies if provided
    if train_accuracies is not None and val_accuracies is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data as CSV
    data = {
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    if train_accuracies is not None:
        data['train_accuracy'] = train_accuracies
        data['val_accuracy'] = val_accuracies
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)

def save_confusion_matrix(y_true, y_pred, class_names, save_dir="./results"):
    """Save confusion matrix visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))

def save_classification_report(y_true, y_pred, class_names, save_dir="./results"):
    """Save classification report"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save as JSON
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save as text
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    
    # Create visualization of classification report
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.iloc[:-3, :-1]  # Remove support column and summary rows
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt='.3f')
    plt.title('Classification Report Heatmap')
    plt.savefig(os.path.join(save_dir, 'classification_report_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_roc_curves(y_true, y_pred_proba, class_names, save_dir="./results"):
    """Save ROC curves for multi-class classification"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to NumPy array if it's a list (fixes slicing error)
    y_pred_proba = np.array(y_pred_proba)

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], linewidth=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save AUC scores
    auc_data = {class_names[i]: roc_auc[i] for i in range(n_classes)}
    with open(os.path.join(save_dir, 'auc_scores.json'), 'w') as f:
        json.dump(auc_data, f, indent=2)


def save_class_distribution(y_true, class_names, save_dir="./results"):
    """Save class distribution visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    unique, counts = np.unique(y_true, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save as CSV
    dist_data = {class_names[i]: counts[i] for i in range(len(unique))}
    df_dist = pd.DataFrame(list(dist_data.items()), columns=['Class', 'Count'])
    df_dist.to_csv(os.path.join(save_dir, 'class_distribution.csv'), index=False)

def save_sample_predictions(images, y_true, y_pred, y_pred_proba, class_names, 
                          num_samples=9, save_dir="./results"):
    """Save sample predictions with images"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        if i >= 9:  # Only show 9 samples
            break
            
        # Display image
        if len(images[idx].shape) == 3:
            axes[i].imshow(images[idx].squeeze(), cmap='gray')
        else:
            axes[i].imshow(images[idx], cmap='gray')
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]] * 100
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(y_true, y_pred, y_pred_proba, class_names, 
                                train_losses=None, val_losses=None, 
                                train_accuracies=None, val_accuracies=None,
                                save_dir="./results"):
    """Generate comprehensive evaluation report with all metrics and visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating comprehensive evaluation report...")
    
    # Save all visualizations
    if train_losses is not None and val_losses is not None:
        save_training_plots(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)
    
    save_confusion_matrix(y_true, y_pred, class_names, save_dir)
    save_classification_report(y_true, y_pred, class_names, save_dir)
    save_roc_curves(y_true, y_pred_proba, class_names, save_dir)
    save_class_distribution(y_true, class_names, save_dir)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Save summary metrics
    summary_metrics = {
        'overall_accuracy': float(accuracy),
        'num_samples': len(y_true),
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    with open(os.path.join(save_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    
    print(f"Comprehensive report saved to {save_dir}")
    print(f"Overall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 3
    class_names = ['Healthy', 'Sick', 'TB']
    
    # Generate dummy data
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Generate dummy training history
    epochs = 10
    train_losses = np.random.exponential(0.5, epochs) + 0.1
    val_losses = train_losses + np.random.normal(0, 0.1, epochs)
    train_accuracies = 1 - train_losses + np.random.normal(0, 0.05, epochs)
    val_accuracies = 1 - val_losses + np.random.normal(0, 0.05, epochs)
    
    # Generate comprehensive report
    generate_comprehensive_report(
        y_true, y_pred, y_pred_proba, class_names,
        train_losses, val_losses, train_accuracies, val_accuracies
    )

