

# import pandas as pd
# import numpy as np
# import torch
# from tqdm import tqdm

# def ndcg_at_k(r, k):
#     r = np.asfarray(r)[:k]
#     if r.size == 0:
#         return 0.0
#     dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
#     idcg = np.sum(sorted(r, reverse=True) / np.log2(np.arange(2, r.size + 2)))
#     return dcg / idcg if idcg > 0 else 0.0

# def hit_at_k(r, k):
#     return 1.0 if np.sum(r[:k]) > 0 else 0.0

# def evaluate_from_file(model, csv_path, output_csv, args):
#     model._net.eval()

#     df = pd.read_csv(csv_path)
#     results = []

#     fixed_score = 0.05
#     skipped_count = 0
#     ndcg_scores = []
#     hit_scores = []

#     max_item_index = model._net.item_embeddings.num_embeddings
#     max_user_index = model._net.user_embeddings.num_embeddings

#     current_user = None
#     step_counter = 0

#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
#         user_id = int(row['user_id'])

#         # Reset step counter on user change
#         if user_id != current_user:
#             current_user = user_id
#             step_counter = 0

#         if user_id >= max_user_index:
#             print(f"⚠️ Skipping unknown user {user_id}")
#             continue

#         true_item = int(row['true_item'])
#         history = eval(row['history'])
#         candidate_items = list(eval(row['candidate_items']))

#         filtered_history = [item for item in history if item < max_item_index]
#         if len(filtered_history) == 0:
#             print(f"⚠️ Skipping user {user_id} due to empty valid history")
#             continue

#         trunc = filtered_history[-args.L:]
#         seq = np.zeros(args.L, dtype=np.int64)
#         seq[-len(trunc):] = trunc

#         user_tensor = torch.LongTensor([[user_id]])
#         seq_tensor = torch.LongTensor([seq])

#         if args.use_cuda:
#             user_tensor = user_tensor.cuda()
#             seq_tensor = seq_tensor.cuda()

#         valid_items = [item for item in candidate_items if item < max_item_index]
#         valid_item_tensor = torch.LongTensor(valid_items)
#         if args.use_cuda:
#             valid_item_tensor = valid_item_tensor.cuda()

#         with torch.no_grad():
#             valid_scores = model._net(seq_tensor, user_tensor, valid_item_tensor, for_pred=True)
#             valid_scores = valid_scores.cpu().numpy().tolist()

#         all_scores = []
#         for item in candidate_items:
#             if item in valid_items:
#                 all_scores.append(valid_scores[valid_items.index(item)])
#             else:
#                 all_scores.append(fixed_score)
#                 skipped_count += 1

#         sorted_indices = np.argsort(all_scores)[::-1]
#         labels = [1 if candidate_items[i] == true_item else 0 for i in sorted_indices]

#         ndcg_scores.append(ndcg_at_k(labels, 10))
#         hit_scores.append(hit_at_k(labels, 10))

#         if true_item in candidate_items:
#             true_index = candidate_items.index(true_item)
#             true_rank = int(np.where(sorted_indices == true_index)[0][0])
#         else:
#             true_rank = -1

#         results.append({
#             'user_id': user_id,
#             'step': step_counter,
#             'true_item': true_item,
#             'candidate_items': candidate_items,
#             'scores': all_scores,
#             'true_rank': true_rank
#         })

#         step_counter += 1

#     pd.DataFrame(results).to_csv(output_csv, index=False)
#     print(f"📊 Saved evaluation results to {output_csv}")
#     print(f"⚠️ Skipped {skipped_count} out-of-range candidate item(s) during evaluation.")

#     metrics = {
#         "ndcg@10": round(np.mean(ndcg_scores), 4),
#         "hit@10": round(np.mean(hit_scores), 4)
#     }

#     print(f"📈 NDCG@10: {metrics['ndcg@10']}, Hit@10: {metrics['hit@10']}")
#     return metrics


import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def ndcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size == 0:
        return 0.0
    dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
    idcg = np.sum(sorted(r, reverse=True) / np.log2(np.arange(2, r.size + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(r, k):
    return 1.0 if np.sum(r[:k]) > 0 else 0.0

def evaluate_from_file(model, csv_path, output_csv, args):
    model._net.eval()

    df = pd.read_csv(csv_path)
    results = []

    fixed_score = 0.05
    skipped_count = 0
    ndcg_scores = []
    hit_scores = []
    all_losses = []

    max_item_index = model._net.item_embeddings.num_embeddings
    max_user_index = model._net.user_embeddings.num_embeddings

    current_user = None
    step_counter = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
        user_id = int(row['user_idx'])

        if user_id != current_user:
            current_user = user_id
            step_counter = 0

        if user_id >= max_user_index:
            print(f"⚠️ Skipping unknown user {user_id}")
            continue

        true_item = int(row['true_item'])
        history = eval(row['history'])
        candidate_items = list(eval(row['candidate_items']))

        filtered_history = [item for item in history if item < max_item_index]
        if len(filtered_history) == 0:
            print(f"⚠️ Skipping user {user_id} due to empty valid history")
            continue

        trunc = filtered_history[-args.L:]
        seq = np.zeros(args.L, dtype=np.int64)
        seq[-len(trunc):] = trunc

        user_tensor = torch.LongTensor([[user_id]])
        seq_tensor = torch.LongTensor([seq])

        if args.use_cuda:
            user_tensor = user_tensor.cuda()
            seq_tensor = seq_tensor.cuda()

        valid_items = [item for item in candidate_items if item < max_item_index]
        valid_item_tensor = torch.LongTensor(valid_items)
        if args.use_cuda:
            valid_item_tensor = valid_item_tensor.cuda()

        with torch.no_grad():
            valid_scores = model._net(seq_tensor, user_tensor, valid_item_tensor, for_pred=True)
            valid_scores = valid_scores.cpu().numpy().tolist()

        all_scores = []
        for item in candidate_items:
            if item in valid_items:
                all_scores.append(valid_scores[valid_items.index(item)])
            else:
                all_scores.append(fixed_score)
                skipped_count += 1

        sorted_indices = np.argsort(all_scores)[::-1]
        labels = [1 if candidate_items[i] == true_item else 0 for i in sorted_indices]

        ndcg_scores.append(ndcg_at_k(labels, 10))
        hit_scores.append(hit_at_k(labels, 10))

        if true_item in candidate_items:
            true_index = candidate_items.index(true_item)
            true_rank = int(np.where(sorted_indices == true_index)[0][0])
            pred_score = all_scores[true_index]

            bce = torch.nn.BCEWithLogitsLoss()
            loss = bce(torch.tensor([pred_score]), torch.tensor([1.0])).item()
        else:
            true_rank = -1
            loss = -1.0

        all_losses.append(loss)

        results.append({
            'user_idx': user_id,
            'step': step_counter,
            'true_item': true_item,
            'candidate_items': candidate_items,
            'scores': all_scores,
            'true_rank': true_rank,
            'loss': loss  # ✅ added
        })

        step_counter += 1

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"📊 Saved evaluation results to {output_csv}")
    print(f"⚠️ Skipped {skipped_count} out-of-range candidate item(s) during evaluation.")

    metrics = {
        "ndcg@10": round(np.mean(ndcg_scores), 4),
        "hit@10": round(np.mean(hit_scores), 4),
        "avg_loss": round(np.mean([l for l in all_losses if l >= 0]), 4)
    }

    print(f"📈 NDCG@10: {metrics['ndcg@10']}, Hit@10: {metrics['hit@10']}, AvgLoss: {metrics['avg_loss']}")
    return metrics



