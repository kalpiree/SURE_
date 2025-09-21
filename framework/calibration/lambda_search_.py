from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from math import sqrt, log

from calibration.risk_estimator import compute_empirical_risk
from calibration.set_constructor_ import construct_prediction_set_at_step

def search_valid_lambda_step0(
    df_model: pd.DataFrame,
    alpha: float = 0.1,
    metric: str = "recall",
    step_t: int = 0,
    lambda_grid: List[float] = None,
    max_pred_set_size: int = None,
    base_utility: float = 1.0,
    delta: float = 0.05
) -> Tuple[float, Dict[Tuple[str, int], List[int]], float]:
    if lambda_grid is None:
        lambda_grid = [round(x, 3) for x in np.linspace(1.0, 0.0, 101)]

    best_lambda = None
    best_risk = None
    best_prediction_sets = None

    df_step = df_model[df_model["step"] == step_t]
    n_users = len(df_step)

    B = 1.0
    lambda_cardinality = len(lambda_grid)
    confidence_correction = B / sqrt(2 * n_users) * sqrt(log((2 * lambda_cardinality) / delta))
    # testing
    # scale_map = {
    #     0.05: 0.20, 0.10: 0.18, 0.15: 0.16, 0.20: 0.14, 0.25: 0.12,
    #     0.30: 0.11, 0.35: 0.10, 0.40: 0.09, 0.45: 0.08, 0.50: 0.07
    # }
    # scale_factor = scale_map.get(round(delta, 2), 1.0)
    # confidence_correction *= scale_factor

    risk_threshold = (1.0 - base_utility) + alpha - confidence_correction

    print(f"\nConfidence correction | Confidence {confidence_correction} | λ⁰-Search | Step {step_t} | α={alpha:.2f} | δ={delta:.2f} | "
          f"BaseUtil={base_utility:.2f} | RiskThreshold={risk_threshold:.4f}")
    print(f"{'λ':>8} | {'Risk':>8} | {'# Users':>9}")

    for lambda_val in lambda_grid:
        prediction_sets = construct_prediction_set_at_step(
            df_model,
            step_t,
            lambda_val,
            max_pred_set_size=max_pred_set_size,
            alpha=alpha
        )
        risk = compute_empirical_risk(df_step, prediction_sets, metric, base_utility=base_utility)
        print(f"{lambda_val:8.3f} | {risk:8.4f} | {n_users:9d}")

        if risk <= risk_threshold:
            best_lambda = lambda_val
            best_risk = risk
            best_prediction_sets = prediction_sets
            print(f"Selected λ = {lambda_val:.3f} with Risk = {risk:.4f}")
            break

    if best_lambda is None:
        best_lambda = 0.0
        best_prediction_sets = construct_prediction_set_at_step(
            df_model,
            step_t,
            best_lambda,
            max_pred_set_size=max_pred_set_size,
            alpha=alpha
        )
        best_risk = compute_empirical_risk(df_step, best_prediction_sets, metric, base_utility=base_utility)
        print(f"No λ found satisfying risk ≤ threshold. Defaulted to λ = 0.000 (Risk = {best_risk:.4f})")

    best_lambda = max(0.0, min(best_lambda, 1.0))

    return best_lambda, best_prediction_sets, best_risk
