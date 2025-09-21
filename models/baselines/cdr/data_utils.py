import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import scipy.sparse as sp
from torch.utils.data import Dataset

class DataVAE(Dataset):
    def __init__(self, data):
        self.data = torch.transpose(data, 0, 1)  # (n_items, n_users)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def data_partition_phase(train_file, eval_file):
    user_train = {}
    user_test = {}

    train_user = []
    train_item = []

    with open(train_file, 'r') as f:
        for line in f:
            u, i = map(int, line.strip().split())
            if u not in user_train:
                user_train[u] = []
            user_train[u].append(i)
            train_user.append(u)
            train_item.append(i)

    eval_df = pd.read_csv(eval_file)
    for idx, row in eval_df.iterrows():
        user_id = int(row['user_idx'])
        true_item = int(row['true_item'])
        user_test[user_id] = true_item

    all_users = set(train_user) | set(user_test.keys())
    n_users = max(all_users) + 1
    all_items = set(train_item)
    for candidates in eval_df['candidate_items'].apply(eval):
        all_items.update(candidates)
    n_items = max(all_items) + 1

    return user_train, user_test, n_users, n_items

def create_train_matrix(user_train, n_users, n_items):
    rows, cols = [], []
    for u, items in user_train.items():
        for i in items:
            rows.append(u)
            cols.append(i)
    train_matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
    return train_matrix

def load_eval_phase(eval_file):
    eval_df = pd.read_csv(eval_file)
    target_items = []
    for idx, row in eval_df.iterrows():
        user_id = int(row['user_idx'])
        true_item = int(row['true_item'])
        target_items.append((user_id, true_item))
    return target_items

