import pandas as pd
import numpy as np
import os

def get_train_val_split_for_fold(df: pd.DataFrame, kfold: int) -> (np.array, np.array, np.array, np.array):
    val_mask = df['kfold'] == kfold
    train_mask = (df['kfold'] != kfold) & (~df['test_data'])
    df.loc[val_mask, 'train_data'] = False
    df.loc[train_mask, 'train_data'] = True
    val_data = df[val_mask]['data'].tolist()
    #val_data = np.expand_dims(np.stack(df[val_mask]['padded_data'].to_numpy()), axis=2)
    val_lbls = np.stack(df[val_mask]['class'].to_numpy())
    train_data = df[train_mask]['data'].tolist()
    #train_data = np.expand_dims(np.stack(df[train_mask]['padded_data'].to_numpy()), axis=2)
    train_lbls = np.stack(df[train_mask]['class'].to_numpy())
    return train_data, train_lbls, val_data, val_lbls

def get_test_data(df: pd.DataFrame) -> (np.array, np.array):
    test_mask = df['test_data']
    test_data = np.stack(df[test_mask]['padded_data'].to_numpy())
    test_lbls = np.stack(df[test_mask]['class'].to_numpy())
    return test_data, test_lbls

def load_pickled_data(dir: str, dataset: str, verbose: bool = True) -> pd.DataFrame:
    if not os.path.exists(dir):
        raise ValueError(f'data dir {dir} DNE')

    expected_pkl_path = os.path.join(dir, f'{dataset}.pkl')

    if not os.path.isfile(expected_pkl_path):
        raise ValueError(f'data dir {expected_pkl_path} DNE')

    df = pd.read_pickle(expected_pkl_path)
    return df
