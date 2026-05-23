import numpy as np


def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    num_samples = X.shape[0]
    shuffled_indices = np.random.permutation(num_samples)
    test_set_size = int(num_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def custom_standard_scaler(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-8
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


class PCA:
    def __init__(self, n_components=None, variance_ratio=None):
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.explained_variance_ratio_ = (S ** 2) / np.sum(S ** 2)

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
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
