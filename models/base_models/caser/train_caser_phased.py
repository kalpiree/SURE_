



import argparse
import os
import numpy as np
import torch

from caser import Caser
from interactions import Interactions
from evaluation_csv import evaluate_from_file
from utils import set_seed, minibatch
from train_helpers import Recommender


def str2bool(s):
    return s.lower() in {'true', '1', 'yes'}

def get_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--phases', type=int, default=5, help='Number of training phases')
    parser.add_argument('--save_dir', type=str, default='caser_runs', help='Base dir to save model/checkpoints')

    # Training
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    # Sequence
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=1)

    # Caser-specific
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='relu')

    return parser.parse_args()


def run_phase(phase_idx, args, train_file, eval_file, save_path, model_ckpt=None):
    phase_name = f'phase{phase_idx}'

    train_data = Interactions(train_file)
    train_data.to_sequence(args.L, args.T)

    recommender = Recommender(
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        neg_samples=args.neg_samples,
        use_cuda=args.use_cuda,
        model_args=args
    )

    if model_ckpt:
        recommender.load(model_ckpt, num_users=train_data.num_users, num_items=train_data.num_items)

    recommender.fit(train_data)

    os.makedirs(save_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, f'{phase_name}_model.pth')
    recommender.save(ckpt_path)
    print(f"Saved: {ckpt_path}")

    out_csv = os.path.join(save_path, f'{phase_name}_eval_output.csv')
    print(f"Evaluating phase {phase_name} using file: {eval_file}")
    metrics = evaluate_from_file(recommender, eval_file, out_csv, args)
    print(f"Eval results saved to: {out_csv}")

    if isinstance(metrics, dict):
        print(f"📈 NDCG@10: {metrics.get('ndcg@10', 'N/A')}, Hit@10: {metrics.get('hit@10', 'N/A')}")

    return ckpt_path


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed, cuda=args.use_cuda)
    root_dir = "processed_datasets"

    for dataset in sorted(os.listdir(root_dir)):
        dataset_path = os.path.join(root_dir, dataset, 'phased_data')
        if not os.path.isdir(dataset_path):
            continue

        for model_id in range(10):
            model_dir = os.path.join(dataset_path, f"model_{model_id}")
            if not os.path.isdir(model_dir):
                continue

            print(f"\n Dataset: {dataset}, Model: model_{model_id}")
            save_path = os.path.join(args.save_dir, f"{dataset}_model{model_id}_caser")
            prev_ckpt = None

            for i in range(args.phases):
                train_file = os.path.join(model_dir, f"train_phase{i}.txt")
                eval_file = os.path.join(model_dir, f"eval_phase{i}.csv")
                prev_ckpt = run_phase(i, args, train_file, eval_file, save_path, model_ckpt=prev_ckpt)

    print("\n All datasets, models, and phases completed.")
