import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


def print_zero_values(df):
    for col in df.columns:
        print(f'{col}: {df.loc[df[col] == 0].shape[0]}')


def print_null_values(df):
    print(df.isnull().any())


def replace_zeros_with_nans(df, *, columns):
    for column in columns:
        df[column] = df[column].replace(0, np.nan)


def fill_nans_with_mean(df, *, columns):
    for column in columns:
        df[column] = df[column].fillna(df[column].mean())


def scale(df, *, preserve_columns=(), ignore_columns=()):
    df_prescaled = df.copy()
    if ignore_columns:
        df_prescaled = df.drop(ignore_columns, axis=1)
    df_scaled = sklearn.preprocessing.scale(df_prescaled)
    df_scaled = pd.DataFrame(df_scaled, columns=df_prescaled.columns)
    for column in preserve_columns:
        df_scaled[column] = df[column]
    return df_scaled


def split_train_test(df, *, target_column, test_size=0.2):
    X = df.loc[:, df.columns != target_column]
    y = df.loc[:, target_column]
    return train_test_split(X, y, test_size=test_size)
