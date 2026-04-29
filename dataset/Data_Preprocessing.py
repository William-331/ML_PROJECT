import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_csv("METABRIC_RNA_Mutation.csv", low_memory=False)   #Tell Pandas not to read the file in chunks to save memory.

# Target variable: overall_survival (0 = deceased, 1 = alive)
y = df['overall_survival'].values

# 1.Extract clinical characteristics and all genetic features
clinical_cols = ['age_at_diagnosis', 'mutation_count', 'tumor_size', 'lymph_nodes_examined_positive']
# All the mutated columns end with '_mut'
mut_cols = [col for col in df.columns if col.endswith('_mut')]
# The remaining columns represent RNA expression levels
rna_cols = [col for col in df.columns[31:] if col not in mut_cols]

# 2.Extract Data Subset
df_clinical = df[clinical_cols].copy()
df_rna = df[rna_cols].copy()
df_mut = df[mut_cols].copy()

# 3.Core Logic of Data Processing
# Clinical and RNA Data: Fill Missing Values with Median
df_clinical = df_clinical.fillna(df_clinical.median())
df_rna = df_rna.fillna(df_rna.median())

# Mutation Data: Binary Conversion (Values equal to the string '0' or the number 0 are set to 0; all other non-zero text is converted to 1)
def binarize_mutation(val):
    if pd.isna(val) or val == 0 or val == '0':
        return 0
    else:
        return 1
# Apply this to all mutant columns
df_mut = df_mut.map(binarize_mutation)

# 4.Concatenate them together column-wise (with axis=1)
X_final = pd.concat([df_clinical, df_rna, df_mut], axis=1)

print(f"Original data dimension: {df.shape}")
print(f"Input dimensions of the model after cleaning: {X_final.shape}")

# Convert a Pandas DataFrame to a pure NumPy matrix
X_matrix = X_final.values

# 5.Random division of the dataset (80% for training, 20% for testing)
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

X_train, X_test, y_train, y_test = custom_train_test_split(X_matrix, y, test_size=0.2)
print(f"Size of the feature matrix of the training set: {X_train.shape}")
print(f"Size of the feature matrix of the test set: {X_test.shape}")

# 6.Feature Scaling (Z-score Scaling)
def custom_standard_scaler(X_train, X_test):
    # Use the mean and standard deviation of the training set to standardize the test set, to prevent data leakage.
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Division by Zero Protection
    std[std == 0] = 1e-8

    # Standardized formula: x' = (x - μ) / σ
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = custom_standard_scaler(X_train, X_test)
















