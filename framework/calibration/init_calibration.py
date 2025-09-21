import pandas as pd
import os
from typing import Dict, Tuple, List

from calibration.lambda_search import search_valid_lambda_step0

def load_processed_model(model_id: int, processed_dir: str = "processed") -> pd.DataFrame:
    """
    Load the preprocessed normalized CSV for a given model ID.
    Assumes filenames like: model_1_normalized.csv
    """
    filepath = os.path.join(processed_dir, f"model_{model_id}_normalized.csv")
    df = pd.read_csv(filepath)

    # Convert stringified lists back to actual Python lists
    df["candidate_items"] = df["candidate_items"].apply(eval)
    df["normalized_scores"] = df["normalized_scores"].apply(eval)

    return df

def calibrate_all_models(
    model_ids: List[int],
    alpha: float = 0.1,
    metric: str = "recall",
    step_t: int = 0,
    processed_dir: str = "processed"
) -> Tuple[
    Dict[int, float],
    Dict[int, Dict[Tuple[str, int], List[int]]],
    Dict[int, Dict[Tuple[str, int], int]]
]:
    """
    Calibrate lambda for all models at step t.

    Returns:
        - lambda_dict: model_id -> lambda^{(0)}
        - prediction_sets: model_id -> {(user_id, step) -> prediction_set}
        - set_sizes: model_id -> {(user_id, step) -> set size}
    """
    lambda_dict = {}
    prediction_sets = {}
    set_sizes = {}

    for model_id in model_ids:
        print(f"Calibrating model {model_id}...")

        try:
            df = load_processed_model(model_id, processed_dir)
        except FileNotFoundError:
            print(f"File for model {model_id} not found in '{processed_dir}'. Skipping.")
            continue

        lambda_val, pred_sets, risk = search_valid_lambda_step0(
            df, alpha=alpha, metric=metric, step_t=step_t
        )

        lambda_dict[model_id] = lambda_val
        prediction_sets[model_id] = pred_sets
        set_sizes[model_id] = {key: len(preds) for key, preds in pred_sets.items()}

        avg_card = sum(set_sizes[model_id].values()) / len(set_sizes[model_id]) if set_sizes[model_id] else 0
        print(f"λ = {lambda_val}, risk = {risk:.4f}, avg set size = {avg_card:.2f}")

    return lambda_dict, prediction_sets, set_sizes
