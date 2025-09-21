

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from calibration.segment_shift import select_segment_start
from calibration.risk_estimator import compute_empirical_risk

def update_lambda(
    current_lambda: float,
    risk_segment: float,
    alpha: float,
    eta_t: float
) -> float:
    return current_lambda - eta_t * (risk_segment - alpha)

def adaptive_update(
    current_lambdas: Dict[int, float],
    loss_traces: Dict[int, Dict[int, float]],
    prediction_sets: Dict[int, Dict[Tuple[int, int], List[int]]],
    df_models: Dict[int, pd.DataFrame],
    current_step: int,
    prev_segment_starts: Dict[int, int],
    alpha: float,
    eta_t: float,
    gamma: float,
    metric: str,
    base_utility: float = 1.0
) -> Tuple[Dict[int, float], Dict[int, int]]:
    updated_lambdas = {}
    new_segment_starts = {}

    print(f"\n[Adaptive Update] at Step {current_step}")
    print(f"   Target risk (alpha): {alpha:.4f}")
    print(f"   Update rate (eta): {eta_t:.4f}")
    print(f"   Segment gamma: {gamma:.4f}")

    for model_id, loss_trace in loss_traces.items():
        print(f"\nModel {model_id}")

        prev_s = prev_segment_starts.get(model_id, 0)

        s_t = select_segment_start(
            losses_all_models=loss_traces,
            model_id=model_id,
            current_step=current_step,
            prev_segment_start=prev_s,
            alpha=alpha,
            gamma=gamma
        )

        print(f"   Segment selected: [{s_t} → {current_step}]")

        risks = []
        df_model = df_models[model_id]

        for t_seg in range(s_t, current_step + 1):
            df_step = df_model[df_model["step"] == t_seg]
            preds = prediction_sets[model_id]
            risk_at_step = compute_empirical_risk(df_step, preds, metric, base_utility=base_utility)
            risks.append(risk_at_step)

        avg_risk = np.mean(risks) if risks else 1.0

        if avg_risk < 0 or avg_risk > 1:
            raise ValueError(f"[Model {model_id}] Average empirical risk = {avg_risk:.4f} out of bounds [0,1] at step {current_step}!")

        print(f"   Average empirical risk over segment = {avg_risk:.4f}")

        current_lambda = current_lambdas[model_id]
        expected_risk = 1.0 - base_utility
        print(f"   Expected risk (base utility) = {expected_risk:.4f}")
        delta_lambda = - eta_t * ((avg_risk - expected_risk) - alpha)
        new_lambda = current_lambda + delta_lambda
        new_lambda = max(0.0, min(new_lambda, 1.0))

        updated_lambdas[model_id] = new_lambda
        new_segment_starts[model_id] = s_t

        print(f"   λ before: {current_lambda:.6f}")
        print(f"   λ update: Δλ = {delta_lambda:.6f}")
        print(f"   λ after:  {new_lambda:.6f}")

    return updated_lambdas, new_segment_starts
