



import sys
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

# ======== Utility Functions ========

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            time_matrix[i][j] = min(span, time_span)
    return time_matrix

# ======== Relation Matrix ========

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(user_train.keys(), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for item in reversed(user_train[user][:-1]):
            time_seq[idx] = 0  # no real timestamp, dummy 0
            idx -= 1
            if idx == -1:
                break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

# ======== Sampler ========

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    user_list = list(user_train.keys())

    def sample(user):
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1]

        idx = maxlen - 1
        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            time_seq[idx] = 0  # dummy
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        while len(one_batch) < batch_size:
            user = random.choice(user_list)
            if len(user_train[user]) <= 1:
                continue
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User, usernum, itemnum, batch_size, maxlen, relation_matrix, self.result_queue, np.random.randint(2e9)))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# ======== Phase-wise Data Loading ========

def data_partition_phase(train_file, eval_file):
    user_train = dict()
    user_test = dict()
    user_set = set()
    item_set = set()

    print('Loading training data from:', train_file)
    with open(train_file, 'r') as f:
        for line in f:
            user_id, item_id = map(int, line.strip().split())
            if user_id not in user_train:
                user_train[user_id] = []
            user_train[user_id].append(item_id)
            user_set.add(user_id)
            item_set.add(item_id)

    print('Loading evaluation data from:', eval_file)
    eval_df = pd.read_csv(eval_file)

    for idx, row in eval_df.iterrows():
        user_id = int(row['user_idx'])
        true_item = int(row['true_item'])
        candidate_items = eval(row['candidate_items'])  # careful: parse as list
        user_test[user_id] = (true_item, candidate_items)
        item_set.update(candidate_items)

    usernum = max(user_set) + 1
    itemnum = max(item_set) + 1

    print('Loaded:', len(user_train), 'users for training,', len(user_test), 'users for testing')
    return user_train, user_test, usernum, itemnum


def evaluate_phase(model, user_train, user_test, args, top_k=10):
    NDCG = 0.0
    HT = 0.0
    MRR = 0.0
    valid_user = 0.0

    users = list(user_test.keys())
    for u in users:
        if u not in user_train or len(user_train[u]) == 0:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        true_item, candidate_items = user_test[u]

        # Dummy time matrix because we have no real timestamps
        time_matrix_dummy = np.zeros((1, args.maxlen, args.maxlen), dtype=np.int32)

        predictions = -model.predict(np.array([u]), np.array([seq]), time_matrix_dummy, np.array(candidate_items))
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < top_k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        MRR += 1.0 / (rank + 1)

    return NDCG / valid_user, HT / valid_user, MRR / valid_user




