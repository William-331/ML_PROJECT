import numpy as np
import time
from models.LogesticRegression import LogisticRegression
from models.SVM import SVM
from utils import calculate_metrics, PCA, cross_validate


def run_all_experiments(X_train, X_test, y_train, y_test):
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
    # 5. SVM — RBF Kernel with CV-based grid search
    # ------------------------------------------------------------------
    print("\n[5/5] SVM RBF (3-fold CV grid search for gamma & lambda) ...")
    t0 = time.time()
    best_g, best_lam = _grid_search_svm_rbf(X_train, y_train)
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
    # Regularization paths
    # ------------------------------------------------------------------
    print("\nComputing regularization paths ...")
    reg_path_data = _compute_regularization_paths(X_train, y_train)

    # ------------------------------------------------------------------
    # Learning curves
    # ------------------------------------------------------------------
    print("Computing learning curves ...")
    lc_data = _compute_learning_curves(X_train, y_train)

    print("Experiments complete.\n")

    return {
        'results': results,
        'cv_results': cv_results,
        'cost_data': cost_data,
        'pca': pca,
        'l1_indices': l1_indices,
        'l1_weights': l1_weights,
        'reg_path_data': reg_path_data,
        'lc_data': lc_data,
    }


def _grid_search_svm_rbf(X_train, y_train):
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
    return best_g, best_lam


def _compute_regularization_paths(X_train, y_train):
    lambda_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    reg_data = {}

    for penalty in ['l2', 'l1']:
        cv_scores = []
        for lam in lambda_values:
            cv = cross_validate(LogisticRegression,
                                dict(learning_rate=0.1, num_iterations=5000,
                                     lambda_param=lam, penalty=penalty),
                                X_train, y_train, cv=5)
            cv_scores.append(cv)
        reg_data[penalty] = (lambda_values, cv_scores)

    return reg_data


def _compute_learning_curves(X_train, y_train):
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
    lc_data = {}

    for name, (model_cls, params) in lc_models.items():
        train_sizes = []
        train_scores = []
        val_scores = []
        for frac in train_fracs:
            n_subset = max(int(X_train.shape[0] * frac), 20)
            indices = np.random.RandomState(42).choice(X_train.shape[0], n_subset, replace=False)
            X_sub, y_sub = X_train[indices], y_train[indices]
            train_sizes.append(n_subset)

            cv = cross_validate(model_cls, params, X_sub, y_sub, cv=5)
            val_scores.append(cv['accuracy'][0])

            model = model_cls(**params)
            model.fit(X_sub, y_sub)
            y_pred_sub = model.predict(X_sub)
            train_scores.append(calculate_metrics(y_sub, y_pred_sub)[0])

        lc_data[name] = (np.array(train_sizes),
                         np.array(train_scores).reshape(-1, 1),
                         np.array(val_scores).reshape(-1, 1))

    return lc_data
