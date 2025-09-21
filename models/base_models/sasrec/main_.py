import os
import glob
import time
import torch
import argparse
import numpy as np
import pandas as pd

from model import SASRec
from utils_ import *
from eval_csv import evaluate_from_file

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='SASRec', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
args = parser.parse_args()

root_dir = 'processed_datasets'
datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for dataset in datasets:
    dataset_path = os.path.join(root_dir, dataset, 'phased_data')

    for model_id in range(10):
        model_dir = os.path.join(dataset_path, f'model_{model_id}')
        if not os.path.isdir(model_dir):
            continue

        train_files = sorted(glob.glob(os.path.join(model_dir, 'train_phase*.txt')))
        phase_count = len(train_files)

        if phase_count == 0:
            continue

        output_dir = f'outputs/{dataset}_model{model_id}_{args.train_dir}'
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
            f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

        prev_ckpt = None
        for phase in range(phase_count):
            train_file = os.path.join(model_dir, f'train_phase{phase}.txt')
            eval_file = os.path.join(model_dir, f'eval_phase{phase}.csv')
            phase_name = f'phase{phase}'

            dataset_obj = data_partition(train_file)
            [user_train, user_valid, user_test, usernum, itemnum] = dataset_obj
            num_batch = (len(user_train) - 1) // args.batch_size + 1

            sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
            model = SASRec(usernum, itemnum, args).to(args.device)

            if prev_ckpt:
                state_dict = torch.load(prev_ckpt, map_location=torch.device(args.device))
                old_item_count = state_dict['item_emb.weight'].shape[0]
                new_item_count = model.item_emb.weight.shape[0]
                if new_item_count > old_item_count:
                    old_weights = state_dict['item_emb.weight']
                    extra = torch.zeros(new_item_count - old_item_count, old_weights.shape[1]).to(old_weights.device)
                    state_dict['item_emb.weight'] = torch.cat([old_weights, extra], dim=0)
                model.load_state_dict(state_dict)

            for name, param in model.named_parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_normal_(param.data)
            model.pos_emb.weight.data[0, :] = 0
            model.item_emb.weight.data[0, :] = 0

            model.train()
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

            for epoch in range(1, args.num_epochs + 1):
                for step in range(num_batch):
                    u, seq, pos, neg = sampler.next_batch()
                    u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

                    pos_logits, neg_logits = model(u, seq, pos, neg)
                    pos_labels = torch.ones_like(pos_logits).to(args.device)
                    neg_labels = torch.zeros_like(neg_logits).to(args.device)

                    optimizer.zero_grad()
                    indices = np.where(pos != 0)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)

                    loss.backward()
                    optimizer.step()

            sampler.close()

            ckpt_path = os.path.join(output_dir, f'{phase_name}_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            prev_ckpt = ckpt_path

            eval_output = os.path.join(output_dir, f'{phase_name}_eval_output.csv')
            evaluate_from_file(model, eval_file, eval_output, args)

print("All datasets and models complete.")
