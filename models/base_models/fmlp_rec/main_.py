import os
import torch
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from glob import glob

from models import FMLPRecModel
from trainers import FMLPRecTrainer
from utils import EarlyStopping, check_path, set_seed, get_dataloder
from evaluation_csv import evaluate_from_file

def load_phase_data(phase_txt):
    df = pd.read_csv(phase_txt, sep=r'\s+', header=None, names=['user', 'item'], dtype={'user': int, 'item': int})
    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1
    train_data = df.values.tolist()
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for u, i in train_data:
        train_mat[u, i] = 1.0
    return train_data, user_num, item_num, train_mat

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--no_filters", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--variance", type=float, default=5)
    return parser.parse_args()

def run_phase(phase_idx, args, train_file, eval_file, save_path, model_ckpt=None):
    print(f"\n===== Phase {phase_idx} =====")
    phase_name = f"phase{phase_idx}"

    train_data, user_num, item_num, _ = load_phase_data(train_file)
    args.item_size = item_num

    os.makedirs(save_path, exist_ok=True)
    args.checkpoint_path = os.path.join(save_path, f"{phase_name}_model.pth")
    args.log_file = os.path.join(save_path, f"{phase_name}.log")

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    model = FMLPRecModel(args)
    device = torch.device("cuda" if args.cuda_condition else "cpu")
    model.to(device)

    if model_ckpt and os.path.exists(model_ckpt):
        print(f"Loading checkpoint from {model_ckpt}")
        state_dict = torch.load(model_ckpt, map_location=device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print(f"Loaded {len(filtered)} compatible parameters")

    user_seq_dict = {}
    for u, i in train_data:
        user_seq_dict.setdefault(u, []).append(i)

    user_ids = sorted(user_seq_dict.keys())
    user_seq = [user_seq_dict[u] for u in user_ids]
    sample_seq = [[0] * 99 for _ in user_ids]

    seq_dic = {
        'user_seq': user_seq,
        'num_users': len(user_ids),
        'sample_seq': sample_seq
    }

    train_loader, eval_loader, test_loader = get_dataloder(args, seq_dic)
    trainer = FMLPRecTrainer(model, train_loader, eval_loader, test_loader, args)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores, _ = trainer.valid(epoch, full_sort=False)
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"Saved checkpoint: {args.checkpoint_path}")

    output_csv = os.path.join(save_path, f"{phase_name}_eval_output.csv")
    metrics = evaluate_from_file(model, eval_file, output_csv, args)
    print(f"Eval results saved to: {output_csv}")
    print(f"NDCG@10: {metrics.get('ndcg@10', 'N/A'):.4f}, Hit@10: {metrics.get('hit@10', 'N/A'):.4f}")

    return args.checkpoint_path

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    root_dir = "processed_datasets"
    for dataset in sorted(os.listdir(root_dir)):
        args.data_name = dataset
        dataset_path = os.path.join(root_dir, dataset, 'phased_data')
        if not os.path.isdir(dataset_path):
            continue

        for model_id in range(10):
            model_dir = os.path.join(dataset_path, f"model_{model_id}")
            if not os.path.isdir(model_dir):
                continue

            print(f"\nDataset: {dataset}, Model: model_{model_id}")

            train_files = sorted(glob(os.path.join(model_dir, "train_phase*.txt")))
            eval_files = sorted(glob(os.path.join(model_dir, "eval_phase*.csv")))
            phase_count = min(len(train_files), len(eval_files))
            if phase_count == 0:
                continue

            save_path = os.path.join("fmlp_runs", f"{dataset}_model{model_id}_fmlprec")
            prev_ckpt = None

            for i in range(phase_count):
                train_file = os.path.join(model_dir, f"train_phase{i}.txt")
                eval_file = os.path.join(model_dir, f"eval_phase{i}.csv")
                prev_ckpt = run_phase(i, args, train_file, eval_file, save_path, model_ckpt=prev_ckpt)

    print("\nAll datasets, models, and phases completed.")
