import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def evaluate_from_file(model, eval_csv_path, output_csv_path, args):
    model.eval()
    device = next(model.parameters()).device

    df = pd.read_csv(eval_csv_path)
    results = []

    ndcg_at_10 = []
    hit_at_10 = []
    all_losses = []

    df['step'] = df.groupby('user_idx').cumcount()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        step = row['step']
        user_id = int(row['user_idx'])
        history = eval(row['history'])
        true_item = int(row['true_item'])
        candidate_items = list(eval(row['candidate_items']))

        history = [item for item in history if item < args.item_size]

        seq = np.zeros(args.max_seq_length, dtype=np.int64)
        trunc = history[-args.max_seq_length:]
        seq[-len(trunc):] = trunc

        seq_tensor = torch.LongTensor([seq]).to(device)

        safe_items = []
        fallback_scores = []
        for item in candidate_items:
            if item < args.item_size:
                safe_items.append(item)
                fallback_scores.append(None)
            else:
                safe_items.append(0)
                fallback_scores.append(0.05)

        item_tensor = torch.LongTensor(safe_items).to(device)

        try:
            with torch.no_grad():
                seq_out = model(seq_tensor)[:, -1, :]
                item_embs = model.item_embeddings(item_tensor)
                raw_scores = torch.matmul(item_embs, seq_out.squeeze(0).T).squeeze().cpu().numpy()

                scores = [
                    fallback_scores[i] if fallback_scores[i] is not None else raw_scores[i]
                    for i in range(len(raw_scores))
                ]
                scores = np.array(scores)

        except Exception:
            scores = np.array([0.05] * len(candidate_items))

        if true_item in candidate_items:
            true_index = candidate_items.index(true_item)
            sorted_indices = np.argsort(scores)[::-1]
            true_rank = int(np.where(sorted_indices == true_index)[0][0])
            pred_score = scores[true_index]

            bce = torch.nn.BCEWithLogitsLoss()
            loss = bce(torch.tensor([pred_score]), torch.tensor([1.0])).item()
        else:
            true_rank = -1
            loss = -1.0

        hit_at_10.append(1 if true_rank != -1 and true_rank < 10 else 0)
        ndcg_at_10.append(1 / np.log2(true_rank + 2) if true_rank != -1 else 0)
        all_losses.append(loss)

        results.append({
            'user_idx': user_id,
            'step': step,
            'true_item': true_item,
            'candidate_items': candidate_items,
            'scores': scores.tolist(),
            'true_rank': true_rank,
            'loss': loss
        })

    pd.DataFrame(results).to_csv(output_csv_path, index=False)

    metrics = {
        "ndcg@10": np.mean(ndcg_at_10),
        "hit@10": np.mean(hit_at_10),
        "avg_loss": np.mean([l for l in all_losses if l >= 0])
    }

    print(f"NDCG@10: {metrics['ndcg@10']:.4f}, Hit@10: {metrics['hit@10']:.4f}, AvgLoss: {metrics['avg_loss']:.4f}")
    return metrics
