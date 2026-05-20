import pandas as pd
import numpy as np

# Random division of the dataset (80% for training, 20% for testing)
def custom_train_test_split(X, y, test_size=0.2, random_state=42):

    # Set a random seed to ensure consistent results each time the program is run
    np.random.seed(random_state)

    # Obtain the total number of rows of the data and generate a shuffled index array
    num_samples = X.shape[0]
    shuffled_indices = np.random.permutation(num_samples)

    # Calculate the number of rows in the test set
    test_set_size = int(num_samples * test_size)

    # Divide the data based on the scrambled indices
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]   

# Feature Scaling (Z-score Scaling)
def custom_standard_scaler(X_train, X_test):
    # Use the mean and standard deviation of the training set to standardize the test set, prevent data leakage.
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Division by Zero Protection
    std[std == 0] = 1e-8

    # Standardized formula: x' = (x - μ) / σ
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


def load_tp53_data(filepath="dataset/METABRIC_RNA_Mutation.csv"):
    # Features: RNA expression only (489 genes after filtering).
    # Target:  tp53_mut (1 = mutated, 0 = wild-type).

    df = pd.read_csv(filepath, low_memory=False)

    rna_cols = [col for col in df.columns[31:] if not col.endswith('_mut')]
    df_rna = df[rna_cols].fillna(df[rna_cols].median())
    X = df_rna.values

    def binarize(v):
        if pd.isna(v) or v == 0 or v == '0':
            return 0
        return 1

    y = df['tp53_mut'].map(binarize).values

    print(f"Data loaded: {X.shape[1]} RNA features, {X.shape[0]} samples")
    print(f"TP53 mutated: {np.sum(y == 1)}, wild-type: {np.sum(y == 0)}")
    return X, y






