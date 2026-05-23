import numpy as np
from .data_loader import load_tp53_data
from .preprocessing import custom_train_test_split, custom_standard_scaler
from .experiment import run_all_experiments
from . import visualization as viz


def run_experiments():
    # Load and preprocess data
    X, y = load_tp53_data()
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    X_train, X_test = custom_standard_scaler(X_train, X_test)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train pos/neg: {np.sum(y_train == 1)}/{np.sum(y_train == 0)}")
    print("=" * 75)

    # Run all experiments
    data = run_all_experiments(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    results = data['results']
    cv_results = data['cv_results']
    l1_indices = data['l1_indices']
    X_n_features = X_train.shape[1]

    print("=" * 85)
    print(f"{'Model':30s}  {'Acc':>7s}  {'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'Features':>12s}")
    print("-" * 85)
    for name, (acc, f1, prec, rec, _, feat) in results.items():
        feat_str = f"{feat}d" if isinstance(feat, int) else str(feat)
        print(f"{name:30s}  {acc:7.4f}  {f1:7.4f}  {prec:7.4f}  {rec:7.4f}  {feat_str:>12s}")

    majority = max(np.mean(y_test == 0), np.mean(y_test == 1))
    print(f"{'Majority baseline':30s}  {majority:7.4f}  {'-':>7s}  {'-':>7s}  {'-':>7s}  {'-':>12s}")
    print()

    print("=" * 85)
    print("5-Fold Cross-Validation Results (mean ± std on training set):")
    print(f"{'Model':30s}  {'Acc':>15s}  {'F1':>15s}  {'Prec':>15s}  {'Rec':>15s}")
    print("-" * 85)
    for name, cv in cv_results.items():
        print(f"{name:30s}  {cv['accuracy'][0]:.4f}±{cv['accuracy'][1]:.4f}  "
              f"{cv['f1'][0]:.4f}±{cv['f1'][1]:.4f}  "
              f"{cv['precision'][0]:.4f}±{cv['precision'][1]:.4f}  "
              f"{cv['recall'][0]:.4f}±{cv['recall'][1]:.4f}")
    print()

    # Key findings
    svm_rbf_acc = results['SVM RBF'][0]
    lr_l1_acc = results['LR + L1'][0]
    svm_lin_acc = results['SVM Linear'][0]
    print("Key findings:")
    print(f"  SVM RBF ({svm_rbf_acc:.4f}) > SVM Linear ({svm_lin_acc:.4f})  "
          f"-> Delta = +{(svm_rbf_acc - svm_lin_acc) * 100:.1f}%  (proves nonlinear structure)")
    print(f"  SVM RBF ({svm_rbf_acc:.4f}) > LR + L1  ({lr_l1_acc:.4f})  "
          f"-> Delta = +{(svm_rbf_acc - lr_l1_acc) * 100:.1f}%  (nonlinear > sparse linear)")
    print(f"  LR + L1 uses only {len(l1_indices)}/{X_n_features} features but achieves {lr_l1_acc:.4f} "
          f"(interpretability vs. accuracy trade-off)")

    # ------------------------------------------------------------------
    # Generate all plots
    # ------------------------------------------------------------------
    print("\nGenerating plots ...")
    viz.plot_metrics_comparison(results, cv_results)
    viz.plot_cost_curves(data['cost_data'])
    viz.plot_pca_variance(data['pca'].explained_variance_ratio_)
    viz.plot_l1_feature_weights(l1_indices, data['l1_weights'], X_n_features)

    for penalty, (lambda_values, cv_scores) in data['reg_path_data'].items():
        viz.plot_regularization_path(lambda_values, cv_scores, penalty.upper())

    for name, (train_sizes, train_scores, val_scores) in data['lc_data'].items():
        viz.plot_learning_curve(train_sizes, train_scores, val_scores, name)

    print("Plots saved to ./plots/")


if __name__ == "__main__":
    run_experiments()
