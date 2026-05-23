import matplotlib
import os as _os

_is_notebook = False
try:
    from IPython import get_ipython
    if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
        _is_notebook = True
except Exception:
    pass

if not _is_notebook:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os


def _save(fig, name):
    os.makedirs('plots', exist_ok=True)
    fig.savefig(f'plots/{name}.png', dpi=150, bbox_inches='tight')
    if not _is_notebook:
        plt.close(fig)


def plot_metrics_comparison(results, cv_results):
    """Grouped bar chart: hold-out metrics + CV metrics side by side."""
    model_names = list(results.keys())
    n = len(model_names)
    x = np.arange(n)
    width = 0.35

    # Hold-out metrics
    holdout_acc = [results[m][0] for m in model_names]
    holdout_f1 = [results[m][1] for m in model_names]

    # CV metrics
    cv_acc_mean = [cv_results[m]['accuracy'][0] for m in model_names]
    cv_acc_std = [cv_results[m]['accuracy'][1] for m in model_names]
    cv_f1_mean = [cv_results[m]['f1'][0] for m in model_names]
    cv_f1_std = [cv_results[m]['f1'][1] for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    bars1 = ax1.bar(x - width/2, holdout_acc, width, label='Hold-out Test', color='steelblue')
    bars2 = ax1.bar(x + width/2, cv_acc_mean, width, yerr=cv_acc_std,
                    label='5-Fold CV', color='coral', capsize=4)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
    ax1.legend()
    ax1.set_ylim(0.5, 1.0)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    # F1
    bars3 = ax2.bar(x - width/2, holdout_f1, width, label='Hold-out Test', color='steelblue')
    bars4 = ax2.bar(x + width/2, cv_f1_mean, width, yerr=cv_f1_std,
                    label='5-Fold CV', color='coral', capsize=4)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Model F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
    ax2.legend()
    ax2.set_ylim(0.5, 1.0)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    _save(fig, '01_metrics_comparison')


def plot_cost_curves(cost_data):
    """Cost descent curves for LR (L1 vs L2) and SVM (Linear vs RBF)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Logistic Regression
    ax1.plot(cost_data['LR + L2'], linewidth=0.8, label='LR + L2 (Ridge)', alpha=0.9)
    ax1.plot(cost_data['LR + L1'], linewidth=0.8, label='LR + L1 (Lasso)', alpha=0.9)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (Cross-Entropy + Regularization)')
    ax1.set_title('Logistic Regression — Cost Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SVM
    ax2.plot(cost_data['SVM Linear'], linewidth=0.8, label='SVM Linear', alpha=0.9)
    ax2.plot(cost_data['SVM RBF'], linewidth=0.8, label='SVM RBF', alpha=0.9)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost (Hinge Loss + L2 Penalty)')
    ax2.set_title('SVM — Cost Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, '02_cost_curves')


def plot_pca_variance(explained_variance_ratio_):
    """Cumulative explained variance vs. number of principal components."""
    cumsum = np.cumsum(explained_variance_ratio_)
    n_comp = np.searchsorted(cumsum, 0.95) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumsum) + 1), cumsum, linewidth=1.5, color='darkgreen')
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='95% threshold')
    ax.axvline(x=n_comp, color='red', linestyle='--', linewidth=1,
               label=f'{n_comp} components')
    ax.fill_between(range(1, n_comp + 1), 0, cumsum[:n_comp], alpha=0.15, color='green')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA — Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(cumsum))
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    _save(fig, '03_pca_variance')


def plot_l1_feature_weights(indices, weights, n_features, top_k=30):
    """Stem plot of top-k L1-selected feature weights."""
    # Take top_k by absolute weight
    sort_order = np.argsort(np.abs(weights))[::-1]
    top_idx = indices[sort_order][:top_k]
    top_w = weights[sort_order][:top_k]

    colors = ['coral' if w < 0 else 'steelblue' for w in top_w]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stem(range(top_k), top_w, linefmt='grey', markerfmt='o', basefmt=' ')
    ax.bar(range(top_k), top_w, color=colors, alpha=0.7)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f'Gene {i}' for i in top_idx], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Weight')
    ax.set_title(f'LR + L1 — Top {top_k} Feature Weights (selected from {n_features} genes)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    _save(fig, '04_l1_feature_weights')


def plot_learning_curve(train_sizes, train_scores, val_scores, model_name):
    """Learning curve: performance vs. number of training samples."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='steelblue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='coral')
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', linewidth=1.5,
            label='Training Accuracy')
    ax.plot(train_sizes, val_mean, 'o-', color='coral', linewidth=1.5,
            label='CV Validation Accuracy')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning Curve — {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    _save(fig, f'learning_curve_{model_name.replace(" ", "_").replace("+", "")}')


def plot_regularization_path(lambda_values, cv_scores, penalty):
    """CV accuracy vs. regularization strength lambda."""
    acc_mean = [s['accuracy'][0] for s in cv_scores]
    acc_std = [s['accuracy'][1] for s in cv_scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(range(len(lambda_values)), acc_mean, yerr=acc_std,
                fmt='o-', capsize=4, color='steelblue', linewidth=1.5)
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels([f'{l:.1e}' if l < 1 else f'{l:.2f}' for l in lambda_values],
                       rotation=30, ha='right', fontsize=8)
    ax.set_xlabel('Lambda (Regularization Strength)')
    ax.set_ylabel('CV Accuracy')
    ax.set_title(f'Regularization Path — {penalty} Penalty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'regularization_path_{penalty}')
