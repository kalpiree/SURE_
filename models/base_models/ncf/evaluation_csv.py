
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def evaluate_from_file(model, csv_path, output_csv, args):
    model.eval()
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    results = []

    hit_10 = 0
    ndcg_10 = 0
    total = 0
    total_loss = 0
    valid_loss_count = 0

    current_user = None
    step_counter = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
        user_id = int(row['user_idx'])

        if user_id != current_user:
            current_user = user_id
            step_counter = 0

        true_item = int(row['true_item'])
        candidate_items = list(eval(row['candidate_items']))

        valid_items = []
        scores = []
        fallback_map = {}

        for i, item in enumerate(candidate_items):
            if item < model.embed_item_GMF.num_embeddings:
                valid_items.append(item)
                fallback_map[i] = False
            else:
                scores.append(0.05)
                fallback_map[i] = True

        if valid_items:
            user_tensor = torch.LongTensor([user_id] * len(valid_items)).to(device)
            item_tensor = torch.LongTensor(valid_items).to(device)

            with torch.no_grad():
                output = model(user_tensor, item_tensor)
                pred_scores = output.view(-1).cpu().numpy().tolist()

            # Fill scores in original order
            pred_iter = iter(pred_scores)
            scores = [next(pred_iter) if not fallback_map[i] else 0.05 for i in range(len(candidate_items))]

        if true_item not in candidate_items:
            true_rank = -1
            loss = -1.0
        else:
            true_index = candidate_items.index(true_item)
            sorted_indices = np.argsort(scores)[::-1]
            true_rank = int(np.where(sorted_indices == true_index)[0][0])
            pred_score = scores[true_index]

            # ✅ Compute loss
            bce = torch.nn.BCEWithLogitsLoss()
            loss = bce(torch.tensor([pred_score]), torch.tensor([1.0])).item()

            total_loss += loss
            valid_loss_count += 1

            if true_rank < 10:
                hit_10 += 1
                ndcg_10 += 1 / np.log2(true_rank + 2)

        results.append({
            'user_idx': user_id,
            'step': step_counter,
            'true_item': true_item,
            'candidate_items': candidate_items,
            'scores': scores,
            'true_rank': true_rank,
            'loss': loss  # ✅ added loss
        })

        step_counter += 1
        total += 1

    # Save CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"📊 Saved evaluation results to {output_csv}")

    # ✅ Return and print metrics
    avg_loss = total_loss / valid_loss_count if valid_loss_count > 0 else 0.0
    metrics = {
        'ndcg@10': ndcg_10 / total,
        'hit@10': hit_10 / total,
        'avg_loss': avg_loss
    }

    print(f"📈 NDCG@10: {metrics['ndcg@10']:.4f}, Hit@10: {metrics['hit@10']:.4f}, AvgLoss: {metrics['avg_loss']:.4f}")
    return metrics
