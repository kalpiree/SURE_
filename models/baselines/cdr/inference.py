import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import models
import data_utils
import evaluate_util

parser = argparse.ArgumentParser(description='CDR Phase-wise Inference')

parser.add_argument('--data_dir', required=True, help='Path to phased data directory')
parser.add_argument('--output_dir', required=True, help='Path where trained models are saved')
parser.add_argument('--save_file', default='inference_results.csv', help='Where to save inference results')

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--mlp_dims', default='[100, 20]')
parser.add_argument('--mlp_p1_dims', default='[100, 200]')
parser.add_argument('--mlp_p2_dims', default='[]')
parser.add_argument('--a_dims', type=int, default=2)
parser.add_argument('--z_dims', type=int, default=2)
parser.add_argument('--c_dims', type=int, default=2)

parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--T', type=int, default=1)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--std', type=float, default=0.1)
parser.add_argument('--w_sigma', type=float, default=0.5)

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--top_k', type=int, default=10)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print(f"Using device: {device}")

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

all_results = []

for model_idx in range(10):
    model_dir = os.path.join(args.data_dir, f'model_{model_idx}')
    print(f"\nInference for Model {model_idx}")

    for phase_idx in range(5):
        print(f"\nPhase {phase_idx}")
        train_file = os.path.join(model_dir, f'train_phase{phase_idx}.txt')
        eval_file = os.path.join(model_dir, f'eval_phase{phase_idx}.csv')
        model_path = os.path.join(args.output_dir, f'model{model_idx}_phase{phase_idx}', 'model.pth')

        user_train, user_test, n_users, n_items = data_utils.data_partition_phase(train_file, eval_file)

        train_matrix = data_utils.create_train_matrix(user_train, n_users, n_items)
        train_matrix = train_matrix[np.newaxis, ...]
        train_tensor = torch.FloatTensor(train_matrix)

        val_dataset = data_utils.DataVAE(train_tensor)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

        mlp_q_dims = [n_items] + eval(args.mlp_dims) + [args.a_dims]
        mlp_p1_dims = [args.a_dims + args.z_dims] + eval(args.mlp_p1_dims) + [args.z_dims]
        mlp_p2_dims = [args.z_dims] + eval(args.mlp_p2_dims) + [n_items]

        model = models.CDR(mlp_q_dims, mlp_p1_dims, mlp_p2_dims,
                           args.c_dims, 0.5, 1, 0, args.T, args.sigma, args.tau, args.std, args.w_sigma).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        eval_df = pd.read_csv(eval_file)
        ground_truth_items = eval_df['true_item'].values.tolist()

        predict_items = []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = torch.transpose(data.to(device), 0, 1)
                recon_batch, _, _, _, _ = model(data)

                _, indices = torch.topk(recon_batch, args.top_k)
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)

        precision, recall, ndcg, mrr = evaluate_util.computeTopNAccuracy(ground_truth_items, predict_items, [args.top_k])
        print(f"Phase {phase_idx} - Precision@{args.top_k}: {precision[0]:.4f}, Recall@{args.top_k}: {recall[0]:.4f}, NDCG@{args.top_k}: {ndcg[0]:.4f}, MRR@{args.top_k}: {mrr[0]:.4f}")

        all_results.append({
            'model_idx': model_idx,
            'phase_idx': phase_idx,
            f'Precision@{args.top_k}': precision[0],
            f'Recall@{args.top_k}': recall[0],
            f'NDCG@{args.top_k}': ndcg[0],
            f'MRR@{args.top_k}': mrr[0],
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(args.output_dir, args.save_file), index=False)

print("Inference Completed and Results Saved!")

