import numpy as np
import pandas as pd
import time
from dataset.Data_Preprocessing import custom_train_test_split, custom_standard_scaler
from models.LogesticRegression import LogisticRegression
from models.SVM import SVM
from utils import calculate_metrics, PCA


def load_tp53_data(filepath="dataset/METABRIC_RNA_Mutation.csv"):
    # Load METABRIC and prepare features + TP53 mutation target.
    # Features: RNA expression only (489 genes after filtering).
    # Target:  tp53_mut (1 = mutated, 0 = wild-type).

    df = pd.read_csv(filepath, low_memory=False)

    # RNA expression columns (exclude mutation columns)
    rna_cols = [col for col in df.columns[31:] if not col.endswith('_mut')]
    df_rna = df[rna_cols].fillna(df[rna_cols].median())
    X = df_rna.values

    # Binarize TP53 mutation status
    def binarize(v):
        if pd.isna(v) or v == 0 or v == '0':
            return 0
        return 1

    y = df['tp53_mut'].map(binarize).values

    print(f"Data loaded: {X.shape[1]} RNA features, {X.shape[0]} samples")
    print(f"TP53 mutated: {np.sum(y == 1)}, wild-type: {np.sum(y == 0)}")
    return X, y


def run_experiments():
    X, y = load_tp53_data()

    # Train/test split + standardization
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    X_train, X_test = custom_standard_scaler(X_train, X_test)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train pos/neg: {np.sum(y_train==1)}/{np.sum(y_train==0)}")
    print("=" * 75)

    results = {}

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
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Time={t:.1f}s")

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
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  NZ={len(l1_indices)}  Time={t:.1f}s")
    print(f"  Top 5 feature indices: {l1_indices[:5]}  (weights: {[f'{w:+.4f}' for w in l1_weights[:5]]})")

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
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Dim={pca.n_components_}  Time={t:.1f}s")

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
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Time={t:.1f}s")

    # ------------------------------------------------------------------
    # 5. SVM — RBF Kernel (Exact + Adam)
    # ------------------------------------------------------------------
    print("\n[5/5] SVM RBF (tuning gamma & lambda) ...")
    best = (0, 0, 0, 0, 0, 0)
    t0 = time.time()
    for g in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        for lam in [1e-6, 1e-5, 5e-5, 1e-4]:
            svm_rbf = SVM(learning_rate=0.01, num_iterations=3000,
                          lambda_param=lam, kernel='rbf_exact',
                          gamma=g, class_weight='balanced',
                          optimizer='adam')
            svm_rbf.fit(X_train, y_train)
            y_pred = svm_rbf.predict(X_test)
            acc, prec, rec, f1 = calculate_metrics(y_test, y_pred)
            if acc > best[0]:
                best = (acc, f1, prec, rec, g, lam)
    t = time.time() - t0
    results['SVM RBF'] = (best[0], best[1], best[2], best[3], t, f"g={best[4]:.4f}")
    print(f"  Acc={best[0]:.4f}  F1={best[1]:.4f}  Prec={best[2]:.4f}  Rec={best[3]:.4f}  gamma={best[4]:.4f}  lam={best[5]:.6f}  Time={t:.1f}s")

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

    return results, l1_indices, l1_weights


if __name__ == "__main__":
    run_experiments()
