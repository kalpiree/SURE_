import os
import torch
import argparse
import time
import numpy as np
import pandas as pd
from trainers import Trainer
from utils import EarlyStopping, set_seed, get_seq_dic, get_dataloder

start_time = time.time()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./processed_datasets/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="goodreads", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--cudaid", default="0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--top_ks", default="10,10,10,10,10", type=str)
    parser.add_argument("--model_name", default="oracle4rec", type=str)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_hidden_layers", default=5, type=int)
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--num_filter_layers", default=1, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--decay_factor", default=0.2, type=float)
    parser.add_argument("--ratio", default=0.75, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--future_max_seq_length", default=10, type=int)
    args = parser.parse_args()

    args.top_ks = list(map(int, args.top_ks.split(',')))
    assert len(args.top_ks) == 5

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.cudaid}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_results = []

    for model_idx in range(10):
        model_dir = os.path.join(args.data_dir, args.data_name, "phased_data", f"model_{model_idx}")
        for phase_idx in range(5):
            args.train_file = os.path.join(model_dir, f"train_phase{phase_idx}.txt")
            args.eval_file = os.path.join(model_dir, f"eval_phase{phase_idx}.csv")
            args.model_dir = os.path.join(args.output_dir, f"{args.data_name}_model{model_idx}_phase{phase_idx}")

            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)

            args.checkpoint_path = os.path.join(args.model_dir, "model.pt")

            seq_dic, max_item = get_seq_dic(args)
            args.item_size = max_item + 1

            train_dataloader, eval_dataloader, test_dataloader, user_seq = get_dataloder(args, seq_dic)

            args.top_k = args.top_ks[phase_idx]
            trainer = Trainer(train_dataloader, eval_dataloader, test_dataloader, user_seq, args)
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)

            for epoch in range(args.epochs):
                trainer.train(epoch)
                scores, _ = trainer.valid_test(epoch, is_valid=True)
                early_stopping(np.array(scores[-1:]), trainer.model, epoch)
                if early_stopping.early_stop:
                    break

            trainer.model.load_state_dict(torch.load(args.checkpoint_path))

            scores, result_info = trainer.valid_test(0, is_valid=False)

            result_row = {
                'dataset_name': args.data_name,
                'model_idx': model_idx,
                'phase_idx': phase_idx,
                f'HR@{args.top_k}': scores[0],
                f'NDCG@{args.top_k}': scores[1],
                'MRR': scores[2],
            }
            all_results.append(result_row)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, f'{args.data_name}_results.csv')
    results_df.to_csv(results_path, index=False)

if __name__ == "__main__":
    main()

end_time = time.time()
print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
