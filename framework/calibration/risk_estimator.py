



import math
from typing import List, Dict, Tuple
import pandas as pd
SUPPORTED_METRICS = {"recall", "ndcg", "mrr"}

def compute_utility(true_item: int, prediction_set: List[int], metric: str = "recall") -> float:
    """
    Compute utility U_M based on the given metric.

    Supported metrics: recall, ndcg, mrr
    """
    if not prediction_set:
        return 0.0

    if metric == "recall":
        return 1.0 if true_item in prediction_set else 0.0

    elif metric == "ndcg":
        try:
            rank = prediction_set.index(true_item)
            return 1.0 / math.log2(rank + 2)  # +2 because rank starts at 0
        except ValueError:
            return 0.0

    elif metric == "mrr":
        try:
            rank = prediction_set.index(true_item)
            return 1.0 / (rank + 1)
        except ValueError:
            return 0.0

    else:
        raise ValueError(f"Unsupported utility metric: {metric}. Supported metrics: {SUPPORTED_METRICS}")

def compute_loss(true_item: int, prediction_set: List[int], metric: str = "recall", base_utility: float = 1.0) -> float:
    """
    Compute loss as (base_utility - actual_utility), clipped at 0.

    Args:
        true_item: the ground truth item.
        prediction_set: ranked list of predicted items.
        metric: recall | ndcg | mrr.
        base_utility: the utility threshold target (e.g., 0.6 instead of 1.0).

    Returns:
        Loss value (non-negative).
    """
    actual_utility = compute_utility(true_item, prediction_set, metric)
    return max(0.0, base_utility - actual_utility)

def compute_empirical_risk(
    df: pd.DataFrame,
    prediction_sets: Dict[Tuple[str, int], List[int]],
    metric: str = "recall",
    base_utility: float = 1.0
) -> float:
    """
    Compute average loss (empirical risk) for all users at a given step.

    Args:
        df: DataFrame filtered to one step.
        prediction_sets: (user, step) -> list of predicted items.
        metric: recall | ndcg | mrr.
        base_utility: the minimum expected utility (e.g., 0.6).

    Returns:
        Average empirical risk over users.
    """
    losses = []
    for _, row in df.iterrows():
        key = (row["user_idx"], row["step"])
        if key not in prediction_sets:
            continue
        pred_set = prediction_sets[key]
        loss = compute_loss(row["true_item"], pred_set, metric, base_utility=base_utility)
        losses.append(loss)

    return sum(losses) / len(losses) if losses else 1.0

