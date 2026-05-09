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


class PCA:
    # Principal Component Analysis implemented from scratch via SVD.
    def __init__(self, n_components=None, variance_ratio=None):
        # n_components: fixed number of components to keep
        # variance_ratio: auto-select components to retain this fraction of total variance (e.g. 0.95)
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.components_ = None    # (k, n_features) — top-k principal axes
        self.mean_ = None           # per-feature mean, used to center data
        self.explained_variance_ratio_ = None
        self.n_components_ = None   # actual number of components after auto-selection

    def fit(self, X):
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD decomposition
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Variance explained by each component
        self.explained_variance_ratio_ = (S ** 2) / np.sum(S ** 2)

        # Decide how many components to keep
        if self.n_components is not None:
            self.n_components_ = min(self.n_components, Vt.shape[0])
        elif self.variance_ratio is not None:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = int(np.searchsorted(cumsum, self.variance_ratio)) + 1
        else:
            self.n_components_ = Vt.shape[0]

        self.components_ = Vt[:self.n_components_]
        return self

    def transform(self, X):
        # Project X onto the top principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)