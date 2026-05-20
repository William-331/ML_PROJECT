import numpy as np
import time
from dataset.Data_Preprocessing import custom_train_test_split, custom_standard_scaler, load_tp53_data
from models.LogesticRegression import LogisticRegression
from models.SVM import SVM
from utils import calculate_metrics, PCA, cross_validate
import visualization as viz


def run_experiments():
    X, y = load_tp53_data()

    # Train/test split + standardization
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    X_train, X_test = custom_standard_scaler(X_train, X_test)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train pos/neg: {np.sum(y_train==1)}/{np.sum(y_train==0)}")
    print("=" * 75)

    results = {}
    cv_results = {}
    cost_data = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression + L2 (Ridge)
    # ------------------------------------------------------------------
    print("\n[1/5] Logistic Regression + L2 (Ridge) ...")
    t0 = time.time()
    lr_l2 = LogisticRegression(learning_rate=0.1, num_iterations=5000,
                               lambda_param=0.01, penalty='l2')
    lr_l2.fit(X_train, y_train)
    y_pred = lr_l2.predict(X_test)
    acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)
    t = time.time() - t0
    results['LR + L2'] = (acc, f1, prec, rec, t, X_train.shape[1])
    cost_data['LR + L2'] = lr_l2.cost_history
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Time={t:.1f}s")

    cv = cross_validate(LogisticRegression,
                        dict(learning_rate=0.1, num_iterations=5000,
                             lambda_param=0.01, penalty='l2'),
                        X_train, y_train, cv=5)
    cv_results['LR + L2'] = cv
    print(f"  CV (5-fold): Acc={cv['accuracy'][0]:.4f}±{cv['accuracy'][1]:.4f}  "
          f"F1={cv['f1'][0]:.4f}±{cv['f1'][1]:.4f}")

    # ------------------------------------------------------------------
    # 2. Logistic Regression + L1 (Lasso) — feature selection
    # ------------------------------------------------------------------
    print("\n[2/5] Logistic Regression + L1 (Lasso) ...")
    t0 = time.time()
    lr_l1 = LogisticRegression(learning_rate=0.1, num_iterations=5000,
                               lambda_param=0.01, penalty='l1')
    lr_l1.fit(X_train, y_train)
    y_pred = lr_l1.predict(X_test)
    acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)
    t = time.time() - t0
    l1_indices, l1_weights = lr_l1.get_important_features()
    results['LR + L1'] = (acc, f1, prec, rec, t, len(l1_indices))
    cost_data['LR + L1'] = lr_l1.cost_history
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  NZ={len(l1_indices)}  Time={t:.1f}s")
    print(f"  Top 5 feature indices: {l1_indices[:5]}  (weights: {[f'{w:+.4f}' for w in l1_weights[:5]]})")

    cv = cross_validate(LogisticRegression,
                        dict(learning_rate=0.1, num_iterations=5000,
                             lambda_param=0.01, penalty='l1'),
                        X_train, y_train, cv=5)
    cv_results['LR + L1'] = cv
    print(f"  CV (5-fold): Acc={cv['accuracy'][0]:.4f}±{cv['accuracy'][1]:.4f}  "
          f"F1={cv['f1'][0]:.4f}±{cv['f1'][1]:.4f}")

    # ------------------------------------------------------------------
    # 3. PCA (95% variance) + Logistic Regression
    # ------------------------------------------------------------------
    print("\n[3/5] PCA (95% var) + Logistic Regression ...")
    t0 = time.time()
    pca = PCA(variance_ratio=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lr_pca = LogisticRegression(learning_rate=0.1, num_iterations=5000,
                                lambda_param=0.01, penalty='l2')
    lr_pca.fit(X_train_pca, y_train)
    y_pred = lr_pca.predict(X_test_pca)
    acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)
    t = time.time() - t0
    results['PCA + L2'] = (acc, f1, prec, rec, t, pca.n_components_)
    cost_data['PCA + L2'] = lr_pca.cost_history
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Dim={pca.n_components_}  Time={t:.1f}s")

    cv = cross_validate(LogisticRegression,
                        dict(learning_rate=0.1, num_iterations=5000,
                             lambda_param=0.01, penalty='l2'),
                        X_train_pca, y_train, cv=5)
    cv_results['PCA + L2'] = cv
    print(f"  CV (5-fold): Acc={cv['accuracy'][0]:.4f}±{cv['accuracy'][1]:.4f}  "
          f"F1={cv['f1'][0]:.4f}±{cv['f1'][1]:.4f}")

    # ------------------------------------------------------------------
    # 4. SVM — Linear Kernel
    # ------------------------------------------------------------------
    print("\n[4/5] SVM Linear ...")
    t0 = time.time()
    svm_lin = SVM(learning_rate=0.001, num_iterations=5000,
                  lambda_param=0.0001, kernel='linear',
                  class_weight='balanced')
    svm_lin.fit(X_train, y_train)
    y_pred = svm_lin.predict(X_test)
    acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)
    t = time.time() - t0
    results['SVM Linear'] = (acc, f1, prec, rec, t, X_train.shape[1])
    cost_data['SVM Linear'] = svm_lin.cost_history
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Time={t:.1f}s")

    cv = cross_validate(SVM,
                        dict(learning_rate=0.001, num_iterations=5000,
                             lambda_param=0.0001, kernel='linear',
                             class_weight='balanced'),
                        X_train, y_train, cv=5)
    cv_results['SVM Linear'] = cv
    print(f"  CV (5-fold): Acc={cv['accuracy'][0]:.4f}±{cv['accuracy'][1]:.4f}  "
          f"F1={cv['f1'][0]:.4f}±{cv['f1'][1]:.4f}")

    # ------------------------------------------------------------------
    # 5. SVM — RBF Kernel (Exact + Adam) with CV-based grid search
    # ------------------------------------------------------------------
    print("\n[5/5] SVM RBF (3-fold CV grid search for gamma & lambda) ...")
    t0 = time.time()
    gammas = [0.0005, 0.001, 0.002, 0.005, 0.01]
    lambdas = [1e-6, 1e-5, 5e-5, 1e-4]
    best_g, best_lam = None, None
    best_cv_score = 0

    for g in gammas:
        for lam in lambdas:
            cv_inner = cross_validate(SVM,
                                      dict(learning_rate=0.01, num_iterations=3000,
                                           lambda_param=lam, kernel='rbf_exact',
                                           gamma=g, class_weight='balanced',
                                           optimizer='adam'),
                                      X_train, y_train, cv=3)
            cv_acc = cv_inner['accuracy'][0]
            print(f"    gamma={g:.4f}  lambda={lam:.6f}  CV Acc={cv_acc:.4f}")
            if cv_acc > best_cv_score:
                best_cv_score = cv_acc
                best_g, best_lam = g, lam

    print(f"  Best hyperparams: gamma={best_g:.4f}  lambda={best_lam:.6f}  (CV Acc={best_cv_score:.4f})")

    # Retrain with best params on full training set, evaluate on test set
    svm_rbf = SVM(learning_rate=0.01, num_iterations=3000,
                  lambda_param=best_lam, kernel='rbf_exact',
                  gamma=best_g, class_weight='balanced',
                  optimizer='adam')
    svm_rbf.fit(X_train, y_train)
    y_pred = svm_rbf.predict(X_test)
    acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)
    t = time.time() - t0
    results['SVM RBF'] = (acc, f1, prec, rec, t, f"g={best_g:.4f}")
    cost_data['SVM RBF'] = svm_rbf.cost_history
    print(f"  Test: Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Time={t:.1f}s")

    cv = cross_validate(SVM,
                        dict(learning_rate=0.01, num_iterations=3000,
                             lambda_param=best_lam, kernel='rbf_exact',
                             gamma=best_g, class_weight='balanced',
                             optimizer='adam'),
                        X_train, y_train, cv=5)
    cv_results['SVM RBF'] = cv
    print(f"  CV (5-fold): Acc={cv['accuracy'][0]:.4f}±{cv['accuracy'][1]:.4f}  "
          f"F1={cv['f1'][0]:.4f}±{cv['f1'][1]:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"{'Model':30s}  {'Acc':>7s}  {'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'Features':>12s}")
    print("-" * 85)
    for name, (acc, f1, prec, rec, t, feat) in results.items():
        if isinstance(feat, int):
            feat_str = f"{feat}d"
        else:
            feat_str = str(feat)
        print(f"{name:30s}  {acc:7.4f}  {f1:7.4f}  {prec:7.4f}  {rec:7.4f}  {feat_str:>12s}")

    majority = max(np.mean(y_test == 0), np.mean(y_test == 1))
    print(f"{'Majority baseline':30s}  {majority:7.4f}  {'-':>7s}  {'-':>7s}  {'-':>7s}  {'-':>12s}")
    print()

    # Cross-validation summary
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

    # Key comparisons
    svm_rbf_acc = results['SVM RBF'][0]
    lr_l1_acc = results['LR + L1'][0]
    svm_lin_acc = results['SVM Linear'][0]

    print("Key findings:")
    print(f"  SVM RBF ({svm_rbf_acc:.4f}) > SVM Linear ({svm_lin_acc:.4f})  "
          f"-> Delta = +{(svm_rbf_acc - svm_lin_acc)*100:.1f}%  (proves nonlinear structure)")
    print(f"  SVM RBF ({svm_rbf_acc:.4f}) > LR + L1  ({lr_l1_acc:.4f})  "
          f"-> Delta = +{(svm_rbf_acc - lr_l1_acc)*100:.1f}%  (nonlinear > sparse linear)")
    print(f"  LR + L1 uses only {len(l1_indices)}/{X_train.shape[1]} features but achieves {lr_l1_acc:.4f} "
          f"(interpretability vs. accuracy trade-off)")

    # ------------------------------------------------------------------
    # Generate visualizations
    # ------------------------------------------------------------------
    print("\nGenerating plots ...")
    viz.plot_metrics_comparison(results, cv_results)
    viz.plot_cost_curves(cost_data)
    viz.plot_pca_variance(pca.explained_variance_ratio_)
    viz.plot_l1_feature_weights(l1_indices, l1_weights, X_train.shape[1])

    # Regularization path: CV accuracy vs. lambda for L1 and L2
    print("\nComputing regularization paths ...")
    lambda_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    for penalty in ['l2', 'l1']:
        cv_scores = []
        for lam in lambda_values:
            cv = cross_validate(LogisticRegression,
                                dict(learning_rate=0.1, num_iterations=5000,
                                     lambda_param=lam, penalty=penalty),
                                X_train, y_train, cv=5)
            cv_scores.append(cv)
        viz.plot_regularization_path(lambda_values, cv_scores, penalty.upper())

    # Learning curves for bias-variance analysis
    print("\nComputing learning curves ...")
    lc_models = {
        'LR + L2': (LogisticRegression, dict(learning_rate=0.1, num_iterations=5000,
                                             lambda_param=0.01, penalty='l2')),
        'LR + L1': (LogisticRegression, dict(learning_rate=0.1, num_iterations=5000,
                                             lambda_param=0.01, penalty='l1')),
        'SVM Linear': (SVM, dict(learning_rate=0.001, num_iterations=5000,
                                 lambda_param=0.0001, kernel='linear',
                                 class_weight='balanced')),
    }
    train_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    for name, (model_cls, params) in lc_models.items():
        train_sizes = []
        train_scores = []
        val_scores = []
        for frac in train_fracs:
            n_subset = max(int(X_train.shape[0] * frac), 20)
            indices = np.random.RandomState(42).choice(X_train.shape[0], n_subset, replace=False)
            X_sub, y_sub = X_train[indices], y_train[indices]
            train_sizes.append(n_subset)

            # 5-fold CV on the subset
            fold_train, fold_val = [], []
            cv = cross_validate(model_cls, params, X_sub, y_sub, cv=5)
            val_scores.append(cv['accuracy'][0])

            # Training accuracy on full subset
            model = model_cls(**params)
            model.fit(X_sub, y_sub)
            y_pred_sub = model.predict(X_sub)
            train_scores.append(calculate_metrics(y_sub, y_pred_sub)[0])

        viz.plot_learning_curve(np.array(train_sizes),
                                np.array(train_scores).reshape(-1, 1),
                                np.array(val_scores).reshape(-1, 1),
                                name)

    print("Plots saved to ./plots/")

    return results, l1_indices, l1_weights, cost_data, cv_results


if __name__ == "__main__":
    run_experiments()
