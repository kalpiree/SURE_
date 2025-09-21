import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def evaluate_from_file(model, csv_path, output_csv, args):
    model.eval()
    df = pd.read_csv(csv_path)
    results = []

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    last_user = None
    user_step = 0

    ndcg_at_10 = []
    hit_at_10 = []
    all_losses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Evaluating'):
        user_id = row['user_idx']
        true_item = row['true_item']
        history = eval(row['history'])
        candidate_items = list(eval(row['candidate_items']))

        if user_id != last_user:
            user_step = 0
            last_user = user_id

        seq = np.zeros([args.maxlen], dtype=np.int32)
        trunc = history[-args.maxlen:]
        if len(trunc) > 0:
            seq[-len(trunc):] = trunc

        itemnum = model.item_num
        valid_items = [item for item in candidate_items if item <= itemnum]
        oov_items = [item for item in candidate_items if item > itemnum]

        scores_dict = {}
        with torch.no_grad():
            if valid_items:
                scores = model.predict(
                    user_ids=np.array([user_id]),
                    log_seqs=np.array([seq]),
                    item_indices=valid_items
                ).cpu().numpy().flatten().tolist()
                scores_dict.update({item: score for item, score in zip(valid_items, scores)})

        scores_dict.update({item: 0.05 for item in oov_items})
        final_scores = [scores_dict[item] for item in candidate_items]

        sorted_indices = np.argsort(final_scores)[::-1]
        if true_item in candidate_items:
            true_idx = candidate_items.index(true_item)
            true_rank = int(np.where(sorted_indices == true_idx)[0][0])
            pred_score = final_scores[true_idx]
            bce = torch.nn.BCEWithLogitsLoss()
            loss = bce(torch.tensor([pred_score]), torch.tensor([1.0])).item()
        else:
            true_rank = -1
            loss = -1.0

        hit_at_10.append(1 if 0 <= true_rank < 10 else 0)
        ndcg_at_10.append(1 / np.log2(true_rank + 2) if true_rank != -1 else 0)
        all_losses.append(loss)

        results.append({
            'user_idx': user_id,
            'step': user_step,
            'true_item': true_item,
            'candidate_items': candidate_items,
            'scores': final_scores,
            'true_rank': true_rank,
            'loss': loss
        })

        user_step += 1

    pd.DataFrame(results).to_csv(output_csv, index=False)

    metrics = {
        "ndcg@10": np.mean(ndcg_at_10),
        "hit@10": np.mean(hit_at_10),
        "avg_loss": np.mean([l for l in all_losses if l >= 0])
    }

    print(f"NDCG@10: {metrics['ndcg@10']:.4f}, Hit@10: {metrics['hit@10']:.4f}, AvgLoss: {metrics['avg_loss']:.4f}")
    return metrics
