

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from visualizations import generate_comprehensive_report

def evaluate_model(labels_path='val_labels.npy', predictions_path='val_predictions.npy', 
                  probabilities_path='val_probabilities.npy'):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    """
    # Load data
    labels = np.load(labels_path)
    predictions = np.load(predictions_path)
    
    # Load probabilities if available
    try:
        probabilities = np.load(probabilities_path)
    except FileNotFoundError:
        print("Probabilities file not found. Some visualizations will be skipped.")
        probabilities = None

    # Class names
    target_names = ['Healthy', 'Sick', 'TB']
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=target_names))

    # Generate comprehensive report with all visualizations
    if probabilities is not None:
        generate_comprehensive_report(
            labels, predictions, probabilities, target_names,
            save_dir="./evaluation_results"
        )
    else:
        # Generate basic report without probabilities
        from visualizations import save_confusion_matrix, save_classification_report, save_class_distribution
        save_confusion_matrix(labels, predictions, target_names, "./evaluation_results")
        save_classification_report(labels, predictions, target_names, "./evaluation_results")
        save_class_distribution(labels, target_names, "./evaluation_results")
    
    print("\nComprehensive evaluation report saved to './evaluation_results' directory")
    
    return accuracy

if __name__ == '__main__':
    evaluate_model()


