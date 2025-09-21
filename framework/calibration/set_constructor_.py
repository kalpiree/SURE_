import pandas as pd
from typing import Dict, Tuple, List

def construct_prediction_set(
    row: pd.Series,
    lambda_t: float,
    max_pred_set_size: int = None,
    alpha: float = None
) -> List[int]:
    candidate_items = row["candidate_items"]
    normalized_scores = row["normalized_scores"]

    threshold = lambda_t

    selected = [(item, score) for item, score in zip(candidate_items, normalized_scores) if score >= threshold]
    selected.sort(key=lambda x: x[1], reverse=True)

    if max_pred_set_size is not None:
        selected = selected[:max_pred_set_size]

    return [item for item, _ in selected]

def construct_prediction_set_at_step(
    df_model: pd.DataFrame,
    step_t: int,
    lambda_t: float,
    max_pred_set_size: int = None,
    alpha: float = None
) -> Dict[Tuple[str, int], List[int]]:
    prediction_sets = {}
    df_at_step = df_model[df_model["step"] == step_t]

    for idx, row in df_at_step.iterrows():
        user_id = row["user_idx"]
        pred_set = construct_prediction_set(
            row,
            lambda_t,
            max_pred_set_size=max_pred_set_size,
            alpha=alpha
        )
        prediction_sets[(user_id, step_t)] = pred_set

    return prediction_sets
