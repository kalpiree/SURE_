import os
import math
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import Dataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model, epoch)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, epoch)
            self.counter = 0

    def save_checkpoint(self, score, model, epoch):
        if self.verbose:
            print(f'Validation score increased. Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
        self.save_epoch = epoch

def build_user_sequences(train_file):
    user_seq_dict = {}
    item_set = set()
    with open(train_file, 'r') as f:
        for line in f:
            user_id, item_id = map(int, line.strip().split())
            if user_id not in user_seq_dict:
                user_seq_dict[user_id] = []
            user_seq_dict[user_id].append(item_id)
            item_set.add(item_id)
    max_item = max(item_set)
    num_users = len(user_seq_dict)
    user_seq = [user_seq_dict[u] for u in sorted(user_seq_dict.keys())]
    return user_seq, max_item, num_users

def load_eval_data(eval_file):
    df = pd.read_csv(eval_file)
    eval_sequences = []
    true_items = []
    candidate_items = []
    for _, row in df.iterrows():
        history = eval(row['history'])
        true_item = int(row['true_item'])
        candidates = eval(row['candidate_items'])
        eval_sequences.append(history)
        true_items.append(true_item)
        candidate_items.append(candidates)
    return eval_sequences, true_items, candidate_items

def get_seq_dic(args):
    user_seq, max_item, num_users = build_user_sequences(args.train_file)
    eval_sequences, true_items, candidate_items = load_eval_data(args.eval_file)
    seq_dic = {
        'user_seq_past': user_seq,
        'num_users': num_users,
        'eval_sequences': eval_sequences,
        'true_items': true_items,
        'candidate_items': candidate_items,
    }
    return seq_dic, max_item

def custom_collate_fn(batch):
    user_ids, input_ids, true_items, candidate_items = zip(*batch)
    user_ids = torch.stack(user_ids)
    input_ids = torch.stack(input_ids)
    true_items = torch.stack(true_items)
    candidate_items = torch.stack(candidate_items)
    return user_ids, input_ids, true_items, candidate_items

def get_dataloder(args, seq_dic):
    train_dataset = Dataset(args, seq_dic['user_seq_past'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    eval_dataset = Dataset(args, seq_dic['eval_sequences'], true_items=seq_dic['true_items'],
                           candidate_items=seq_dic['candidate_items'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    test_dataset = Dataset(args, seq_dic['eval_sequences'], true_items=seq_dic['true_items'],
                           candidate_items=seq_dic['candidate_items'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    return train_dataloader, eval_dataloader, test_dataloader, seq_dic['user_seq_past']

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)
