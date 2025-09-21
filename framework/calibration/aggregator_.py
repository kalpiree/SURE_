

import random
import numpy as np
from typing import Dict, List, Tuple, Set

def aggregate_prediction_sets(
    user_id: int,
    step: int,
    model_sets: Dict[int, Dict[Tuple[int, int], List[int]]],
    weights: Dict[int, float],
    candidate_items: Set[int],
    seed: int = None,
    alpha: float = 0.1
) -> List[int]:
    if seed is not None:
        random.seed(seed)

    base_threshold = 0.5

    alpha_values = [0.05, 0.07, 0.1, 0.12, 0.15]
    shift = 0.0
    if alpha in alpha_values:
        index = alpha_values[::-1].index(alpha)
        shift = index * 0.02

    threshold = base_threshold + shift

    item_support = {}
    active_models = {}

    for model_id, pred_dict in model_sets.items():
        pred_set = pred_dict.get((user_id, step), [])
        if pred_set:
            active_models[model_id] = pred_set

    if not active_models:
        return []

    active_weight_sum = sum(weights[m] for m in active_models)
    if active_weight_sum == 0:
        normalized_weights = {m: 1.0 / len(active_models) for m in active_models}
    else:
        normalized_weights = {m: weights[m] / active_weight_sum for m in active_models}

    for item in candidate_items:
        support = sum(
            normalized_weights[model_id]
            for model_id, pred_set in active_models.items()
            if item in pred_set
        )
        item_support[item] = support

    final_prediction_set = [item for item, support in item_support.items() if support > threshold]

    return final_prediction_set

def update_weights(
    set_sizes: Dict[int, Dict[Tuple[int, int], int]],
    current_step: int,
    eta: float
) -> Dict[int, float]:
    L = len(set_sizes)
    cumulative_sizes = {}
    exp_weights = {}

    print(f"\nUpdating weights at step {current_step}:")

    for model_id, user_step_sizes in set_sizes.items():
        user_ids = {user for (user, _) in user_step_sizes}
        num_users = len(user_ids)

        total_size = sum(
            size for (user, step), size in user_step_sizes.items()
            if step <= current_step
        )

        normalized_size = total_size / ((num_users * (current_step + 1)) + 1e-8)
        cumulative_sizes[model_id] = normalized_size
        print(f"   - Model {model_id} | Cumulative size: {total_size} | Users: {num_users} | Normalized: {normalized_size:.4f}")

    for model_id in cumulative_sizes:
        SCALE = 25
        scaled = cumulative_sizes[model_id] / SCALE
        exp_weights[model_id] = np.exp(-eta * scaled)

    total_weight = sum(exp_weights.values())
    print(f"   Raw exp weights: {exp_weights}")
    print(f"   Total exp weight: {total_weight:.4f}")

    if total_weight == 0 or not np.isfinite(total_weight):
        print("   Total weight is zero or invalid! Using uniform weights.")
        return {model_id: 1.0 / L for model_id in cumulative_sizes}

    weights = {
        model_id: weight / total_weight
        for model_id, weight in exp_weights.items()
    }

    print(f"   New Weights: {weights}")
    return weights
