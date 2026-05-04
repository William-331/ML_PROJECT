import numpy as np

def calculate_metrics(y_true, y_pred):
    # Calculate classification metrics from scratch.
    # y_true: Ground truth labels 
    # y_pred: Predicted labels 

    # True Positives
    TP = np.sum((y_pred == 1) & (y_true == 1))
    
    # True Negatives
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    # False Positives
    FP = np.sum((y_pred == 1) & (y_true == 0))
    
    # False Negatives
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    # Accuracy
    accuracy = (TP + TN) / len(y_true)
    
    # Precision
    # 1e-9: Prevent error from occurring when the denominator is 0
    precision = TP / (TP + FP + 1e-9)
    
    # Recall
    recall = TP / (TP + FN + 1e-9)
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    return accuracy, precision, recall, f1_score