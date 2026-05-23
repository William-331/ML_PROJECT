import pandas as pd
import numpy as np


def load_tp53_data(filepath="dataset/METABRIC_RNA_Mutation.csv"):
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
