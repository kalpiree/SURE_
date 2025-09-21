import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import models
import data_utils
import evaluate_util

start_time = time.time()

parser = argparse.ArgumentParser(description='CDR Phase-wise Training')

parser.add_argument('--data_dir', required=True)
parser.add_argument('--output_dir', default='./output_cdr')

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=0.0)

parser.add_argument('--mlp_dims', default='[100, 20]')
parser.add_argument('--mlp_p1_dims', default='[100, 200]')
parser.add_argument('--mlp_p2_dims', default='[]')
parser.add_argument('--a_dims', type=int, default=2)
parser.add_argument('--z_dims', type=int, default=2)
parser.add_argument('--c_dims', type=int, default=2)

parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--T', type=int, default=1)
parser.add_argument('--sigma', type=float, default=0.1)

parser.add_argument('--total_anneal_steps', type=int, default=0)
parser.add_argument('--lam1', type=float, default=0.2)
parser.add_argument('--lam2', type=float, default=0.1)
parser.add_argument('--lam3', type=float, default=0.1)

parser.add_argument('--regs', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--bn', type=int, default=1)

parser.add_argument('--std', type=float, default=0.1)
parser.add_argument('--w_sigma', type=float, default=0.5)

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--top_ks', default='25,25,25,25,25')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--add_T', type=int, default=2)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

args.top_ks = list(map(int, args.top_ks.split(',')))
assert len(args.top_ks) == 5

print(args)
print(f"Using device: {device}")

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

def train_one_epoch(model, train_loader, optimizer, epoch, update_count):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = torch.transpose(data.to(device), 0, 1)

        if args.total_anneal_steps > 0:
            anneal = min(args.lam1, update_count / args.total_anneal_steps)
        else:
            anneal = args.lam1

        optimizer.zero_grad()
        _, rec_xT, mu_T, logvar_T, reg_loss = model(data)

        rec_loss, env_variance = models.loss_variance(model, rec_xT, data, mu_T, logvar_T, anneal)
        var_loss = torch.sum(env_variance)
        penalty_w = torch.sum(models.loss_penalty(model.W_CK / model.sigma)) + \
                    torch.sum(models.loss_penalty(model.W_CD / model.sigma))

        total_loss = rec_loss + args.lam2 * penalty_w + args.lam3 * var_loss + args.regs * reg_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100, norm_type=2)
        optimizer.step()

        update_count += 1
    return update_count

def evaluate(model, val_loader, ground_truth_items, candidate_items_list, top_k):
    model.eval()
    predict_items = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = torch.transpose(data.to(device), 0, 1)
            recon_batch, _, _, _, _ = model(data)

            batch_size = recon_batch.shape[0]
            for i in range(batch_size):
                candidates = candidate_items_list[batch_idx * args.batch_size + i]
                mask = torch.full_like(recon_batch[i], float('-inf'))
                mask[candidates] = recon_batch[i][candidates]
                recon_batch[i] = mask

            _, indices = torch.topk(recon_batch, top_k)
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    precision, recall, ndcg, mrr = evaluate_util.computeTopNAccuracy(ground_truth_items, predict_items, [top_k])
    return precision[0], recall[0], ndcg[0], mrr[0]

update_count = 0
all_results = []

for model_idx in range(10):
    model_dir = os.path.join(args.data_dir, f'model_{model_idx}')
    print(f"\nTraining Model {model_idx}")

    for phase_idx in range(5):
        print(f"\nPhase {phase_idx}")
        train_file = os.path.join(model_dir, f'train_phase{phase_idx}.txt')
        eval_file = os.path.join(model_dir, f'eval_phase{phase_idx}.csv')

        user_train, user_test, n_users, n_items = data_utils.data_partition_phase(train_file, eval_file)
        train_matrix = data_utils.create_train_matrix(user_train, n_users, n_items)
        train_matrix = train_matrix.toarray()[np.newaxis, ...]

        print(f'Users: {n_users}, Items: {n_items}')
        print(f'Avg sequence length: {np.sum(train_matrix) / n_users:.2f}')

        train_tensor = torch.FloatTensor(train_matrix)
        train_dataset = data_utils.DataVAE(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

        mlp_q_dims = [n_items] + eval(args.mlp_dims) + [args.a_dims]
        mlp_p1_dims = [args.a_dims + args.z_dims] + eval(args.mlp_p1_dims) + [args.z_dims]
        mlp_p2_dims = [args.z_dims] + eval(args.mlp_p2_dims) + [n_items]

        model = models.CDR(mlp_q_dims, mlp_p1_dims, mlp_p2_dims,
                           args.c_dims, args.dropout, args.bn,
                           args.regs, args.T, args.sigma, args.tau, args.std, args.w_sigma).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        eval_df = pd.read_csv(eval_file)
        ground_truth_items = eval_df['true_item'].tolist()
        candidate_items_list = eval_df['candidate_items'].apply(eval).tolist()

        for epoch in range(1, args.epochs + 1):
            update_count = train_one_epoch(model, train_loader, optimizer, epoch, update_count)

        top_k = args.top_ks[phase_idx]
        val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        prec, rec, ndcg, mrr = evaluate(model, val_loader, ground_truth_items, candidate_items_list, top_k)

        print(f"Phase {phase_idx} - Precision@{top_k}: {prec:.4f}, Recall@{top_k}: {rec:.4f}, NDCG@{top_k}: {ndcg:.4f}, MRR@{top_k}: {mrr:.4f}")

        save_dir = os.path.join(args.output_dir, f'model{model_idx}_phase{phase_idx}')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

        all_results.append({
            'model_idx': model_idx,
            'phase_idx': phase_idx,
            f'Precision@{top_k}': prec,
            f'Recall@{top_k}': rec,
            f'NDCG@{top_k}': ndcg,
            f'MRR@{top_k}': mrr,
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(args.output_dir, 'final_results.csv'), index=False)

print("Training and Evaluation Complete!")
end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
