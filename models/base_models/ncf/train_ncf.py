import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import scipy.sparse as sp
from glob import glob

from model import NCF
from data_utils import NCFData
from evaluation_csv import evaluate_from_file
from utils import set_seed

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def load_phase_data(phase_txt):
    try:
        df = pd.read_csv(phase_txt, sep=',', header=None, names=['user', 'item'], dtype={'user': int, 'item': int})
    except:
        try:
            df = pd.read_csv(phase_txt, sep='\t', header=None, names=['user', 'item'], dtype={'user': int, 'item': int})
        except:
            df = pd.read_csv(phase_txt, sep=r'\s+', engine='python', header=None, names=['user', 'item'], dtype={'user': int, 'item': int})
    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1
    train_data = df.values.tolist()
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for u, i in train_data:
        train_mat[u, i] = 1.0
    return train_data, user_num, item_num, train_mat

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_ng', type=int, default=4)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--factor_num', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=10)
    return parser.parse_args()

def run_phase(phase_idx, args, train_file, eval_file, output_dir, model_ckpt=None):
    print(f"\n===== Phase {phase_idx} =====")
    phase_name = f'phase{phase_idx}'

    train_data, user_num, item_num, train_mat = load_phase_data(train_file)
    train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, is_training=True)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, 'NeuMF-end', None, None)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    if model_ckpt:
        print(f"Loading previous checkpoint: {model_ckpt}")
        state_dict = torch.load(model_ckpt, map_location=device)
        model_state = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
        model_state.update(filtered_dict)
        model.load_state_dict(model_state)
        print(f"Loaded {len(filtered_dict)} / {len(model_state)} matching parameters.")

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        train_loader.dataset.ng_sample()
        total_loss = 0.0
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)
            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f'{phase_name}_model.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    out_csv = os.path.join(output_dir, f'{phase_name}_eval_output.csv')
    metrics = evaluate_from_file(model, eval_file, out_csv, args)
    print(f"Eval results saved to: {out_csv}")
    print(f"NDCG@10: {metrics.get('ndcg@10', 'N/A'):.4f}, Hit@10: {metrics.get('hit@10', 'N/A'):.4f}, AvgLoss: {metrics.get('avg_loss', 'N/A'):.4f}")
    return ckpt_path

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed, cuda=args.use_cuda)

    root = 'processed_datasets'
    for dataset in sorted(os.listdir(root)):
        dataset_dir = os.path.join(root, dataset, 'phased_data')
        if not os.path.isdir(dataset_dir):
            continue

        for model_id in range(10):
            model_dir = os.path.join(dataset_dir, f'model_{model_id}')
            if not os.path.isdir(model_dir):
                continue

            print(f"\nDataset: {dataset}, Model: model_{model_id}")

            train_files = sorted(glob(os.path.join(model_dir, 'train_phase*.txt')))
            eval_files = sorted(glob(os.path.join(model_dir, 'eval_phase*.csv')))
            phase_count = min(len(train_files), len(eval_files))

            if phase_count == 0:
                continue

            output_dir = os.path.join('outputs_ncf', f'{dataset}_model{model_id}_ncf')
            prev_ckpt = None

            for phase in range(phase_count):
                train_file = os.path.join(model_dir, f'train_phase{phase}.txt')
                eval_file = os.path.join(model_dir, f'eval_phase{phase}.csv')
                prev_ckpt = run_phase(phase, args, train_file, eval_file, output_dir, model_ckpt=prev_ckpt)

    print("\nAll datasets, models, and phases completed.")
