# import os
# import time
# import torch
# import pickle
# import argparse

# from model import TiSASRec
# from tqdm import tqdm
# from utils import *

# def str2bool(s):
#     if s not in {'false', 'true'}:
#         raise ValueError('Not a valid boolean string')
#     return s == 'true'

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True)
# parser.add_argument('--train_dir', required=True)
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=201, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.2, type=float)
# parser.add_argument('--l2_emb', default=0.00005, type=float)
# parser.add_argument('--device', default='cpu', type=str)
# parser.add_argument('--inference_only', default=False, type=str2bool)
# parser.add_argument('--state_dict_path', default=None, type=str)
# parser.add_argument('--time_span', default=256, type=int)

# args = parser.parse_args()
# if not os.path.isdir(args.dataset + '_' + args.train_dir):
#     os.makedirs(args.dataset + '_' + args.train_dir)
# with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

# dataset = data_partition(args.dataset)
# [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
# num_batch = len(user_train) // args.batch_size
# cc = 0.0
# for u in user_train:
#     cc += len(user_train[u])
# print('average sequence length: %.2f' % (cc / len(user_train)))

# f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

# try:
#     relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
# except:
#     relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
#     pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))

# sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
# model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)

# for name, param in model.named_parameters():
#     try:
#         torch.nn.init.xavier_uniform_(param.data)
#     except:
#         pass # just ignore those failed init layers

# model.train() # enable model training

# epoch_start_idx = 1
# if args.state_dict_path is not None:
#     try:
#         model.load_state_dict(torch.load(args.state_dict_path))
#         tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
#         epoch_start_idx = int(tail[:tail.find('.')]) + 1
#     except:
#         print('failed loading state_dicts, pls check file path: ', end="")
#         print(args.state_dict_path)

# if args.inference_only:
#     model.eval()
#     t_test = evaluate(model, dataset, args)
#     print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

# bce_criterion = torch.nn.BCEWithLogitsLoss()
# adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

# T = 0.0
# t0 = time.time()

# for epoch in range(epoch_start_idx, args.num_epochs + 1):
#     if args.inference_only: break # just to decrease identition
#     for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
#         u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch() # tuples to ndarray
#         u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
#         time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
#         pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)
#         pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
#         # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
#         adam_optimizer.zero_grad()
#         indices = np.where(pos != 0)
#         loss = bce_criterion(pos_logits[indices], pos_labels[indices])
#         loss += bce_criterion(neg_logits[indices], neg_labels[indices])
#         for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
#         for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
#         for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
#         for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
#         for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
#         loss.backward()
#         adam_optimizer.step()
#         print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

#     if epoch % 20 == 0:
#         model.eval()
#         t1 = time.time() - t0
#         T += t1
#         print('Evaluating', end='')
#         t_test = evaluate(model, dataset, args)
#         t_valid = evaluate_valid(model, dataset, args)
#         print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
#                 % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

#         f.write(str(t_valid) + ' ' + str(t_test) + '\n')
#         f.flush()
#         t0 = time.time()
#         model.train()

#     if epoch == args.num_epochs:
#         folder = args.dataset + '_' + args.train_dir
#         fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
#         fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
#         torch.save(model.state_dict(), os.path.join(folder, fname))

# f.close()
# sampler.close()
# print("Done")

import os
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import TiSASRec
from utils import *
from utils import WarpSampler
start_time = time.time()

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()

# General arguments
parser.add_argument('--data_dir', required=True, type=str)  # e.g., ./processed_datasets/goodreads/phased_data/
parser.add_argument('--output_dir', default='output/', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--time_span', default=256, type=int)

# NEW: Top-K values per phase
parser.add_argument('--top_ks', default="20,20,20,20,20", type=str, help="Comma-separated Top-K values for each phase")

args = parser.parse_args()

# Parse top_ks
args.top_ks = list(map(int, args.top_ks.split(',')))
assert len(args.top_ks) == 5, "You must provide exactly 5 top_k values for the 5 phases."

os.makedirs(args.output_dir, exist_ok=True)

all_results = []

# Loop through models
for model_idx in range(10):
    model_dir = os.path.join(args.data_dir, f'model_{model_idx}')
    
    print(f"\n===== Processing Model {model_idx} =====")

    # Loop through phases
    for phase_idx in range(5):
        print(f"\n--- Phase {phase_idx} ---")

        train_file = os.path.join(model_dir, f'train_phase{phase_idx}.txt')
        eval_file = os.path.join(model_dir, f'eval_phase{phase_idx}.csv')
        save_dir = os.path.join(args.output_dir, f'model{model_idx}_phase{phase_idx}')
        os.makedirs(save_dir, exist_ok=True)

        user_train, user_test, usernum, itemnum = data_partition_phase(train_file, eval_file)

        print('Average sequence length: %.2f' % (
            sum(len(user_train[u]) for u in user_train) / len(user_train)
        ))

        # Prepare relation matrix
        rel_matrix_path = os.path.join(save_dir, f'relation_matrix_model{model_idx}_phase{phase_idx}.pickle')
        if os.path.exists(rel_matrix_path):
            relation_matrix = pickle.load(open(rel_matrix_path, 'rb'))
        else:
            relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
            pickle.dump(relation_matrix, open(rel_matrix_path, 'wb'))

        sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix,
                               batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

        model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)
        model.apply(lambda module: torch.nn.init.xavier_uniform_(module.weight) if hasattr(module, 'weight') and module.weight.dim() > 1 else None)

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        model.train()

        for epoch in range(1, args.num_epochs + 1):
            for step in range(len(user_train) // args.batch_size):
                u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)

                pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)

                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + \
                       bce_criterion(neg_logits[indices], neg_labels[indices])

                # L2 regularization
                for param in model.parameters():
                    loss += args.l2_emb * torch.norm(param)

                loss.backward()
                adam_optimizer.step()

        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

        # Evaluate
        model.eval()
        top_k_this_phase = args.top_ks[phase_idx]
        ndcg, hr, mrr = evaluate_phase(model, user_train, user_test, args, top_k=top_k_this_phase)

        # print('Phase %d Results - NDCG@%d: %.4f HR@%d: %.4f' % 
        #       (phase_idx, top_k_this_phase, ndcg, top_k_this_phase, hr))
        print('Phase %d Results - NDCG@%d: %.4f HR@%d: %.4f MRR: %.4f' % 
            (phase_idx, top_k_this_phase, ndcg, top_k_this_phase, hr, mrr))



        result_row = {
            'model_idx': model_idx,
            'phase_idx': phase_idx,
            f'NDCG@{top_k_this_phase}': ndcg,
            f'HR@{top_k_this_phase}': hr,
            'MRR': mrr
        }
        all_results.append(result_row)

        sampler.close()

# Save all results
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(args.output_dir, 'final_results.csv'), index=False)

print("\n Training and Evaluation Complete! Results saved.")
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal Execution Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")



