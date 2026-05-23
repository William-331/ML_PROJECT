import numpy as np


def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return accuracy, precision, recall, f1_score


class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        n = X.shape[0]
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[:n % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop


def cross_validate(model_class, model_params, X, y, cv=5, metrics_func=None):
    if metrics_func is None:
        metrics_func = calculate_metrics

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

    for train_idx, val_idx in kf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        acc, prec, rec, f1 = metrics_func(y_val_fold, y_pred)
        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_rec.append(rec)
        fold_f1.append(f1)

    return {
        'accuracy': (np.mean(fold_acc), np.std(fold_acc)),
        'precision': (np.mean(fold_prec), np.std(fold_prec)),
        'recall': (np.mean(fold_rec), np.std(fold_rec)),
        'f1': (np.mean(fold_f1), np.std(fold_f1)),
    }
