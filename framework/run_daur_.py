import os, math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from calibration.set_constructor_ import construct_prediction_set_at_step
from calibration.aggregator_ import aggregate_prediction_sets, update_weights
from calibration.risk_estimator import compute_empirical_risk, compute_loss
from calibration.adaptive_loop import adaptive_update
from calibration.segment_shift import select_segment_start

def run_daur(
    phase_files: List[str],
    model_ids: List[int],
    initial_lambdas: Dict[int, float],
    initial_weights: Dict[int, float],
    df_models: Dict[int, pd.DataFrame],
    alpha: float,
    eta: float,
    gamma: float,
    metric: str = "recall",
    save_outputs: bool = True,
    output_dir: str = "outputs",
    frozen_inference: bool = False,
    max_pred_set_size: int = None,
    base_utility: float = 1.0,
    delta: float = 0.05
):
    lambdas = {0: initial_lambdas.copy()}
    weights = {0: initial_weights.copy()}
    segment_starts = {0: {m: 0 for m in model_ids}}
    losses = {m: {} for m in model_ids}
    model_risks = {m: {} for m in model_ids}
    prediction_sets = {m: {} for m in model_ids}
    set_sizes = {m: {} for m in model_ids}
    ensemble_sets = {}
    ensemble_eval = []
    diag_rows = []

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    current_step = 0
    last_phase_file = phase_files[-1] if phase_files else None
    df_model_ref = None
    steps_to_run = []

    if phase_files:
        for phase_file in phase_files:
            phase_df = pd.read_csv(phase_file)
            phase_df["candidate_items"] = phase_df["candidate_items"].apply(eval)
            phase_df["normalized_scores"] = phase_df["normalized_scores"].apply(eval)
            df_model_ref = phase_df
            steps_to_run.extend(sorted(df_model_ref["step"].unique()))
    else:
        df_model_ref = df_models[model_ids[0]]
        steps_to_run = sorted(df_model_ref["step"].unique())

    for m in model_ids:
        df_models[m]["normalized_loss"] = df_models[m]["loss"]

    skip_steps = set(range(4, 50, 5))

    for step in steps_to_run:
        if current_step == 0 and max_pred_set_size is not None:
            pass

        for m in model_ids:
            df_model = df_models[m]
            lambda_t = initial_lambdas[m] if current_step == 0 else lambdas[current_step][m]

            preds = construct_prediction_set_at_step(
                df_model,
                step,
                lambda_t,
                max_pred_set_size=max_pred_set_size,
                alpha=alpha
            )
            prediction_sets[m].update(preds)

            for k, v in preds.items():
                set_sizes[m][k] = len(v)

            df_step = df_model[df_model["step"] == step]
            risk_val = compute_empirical_risk(df_step, preds, metric, base_utility=base_utility)
            avg_loss = df_step["normalized_loss"].mean()

            model_risks[m][current_step] = risk_val
            losses[m][current_step] = avg_loss

        users = df_model_ref[df_model_ref["step"] == step]["user_idx"].unique()
        candidate_items = {
            u: set(it for row in df_model_ref[(df_model_ref["user_idx"]==u)&(df_model_ref["step"]==step)]["candidate_items"] for it in row)
            for u in users
        }

        ensemble_at_t = {
            (u, current_step): aggregate_prediction_sets(
                user_id=u,
                step=current_step,
                model_sets=prediction_sets,
                weights=weights[current_step],
                candidate_items=candidate_items[u],
                seed=current_step)
            for u in users
        }
        ensemble_sets[current_step] = ensemble_at_t

        if current_step in skip_steps:
            df_step = df_model_ref[df_model_ref["step"] == step]

            m_risk, m_size = {}, {}
            for m in model_ids:
                preds_m = {k: v for k, v in prediction_sets[m].items() if k[1] == current_step}
                risks, sizes = [], []
                for u in users:
                    key = (u, current_step)
                    if key not in preds_m:
                        continue
                    true_item = df_step[df_step["user_idx"] == u]["true_item"].values[0]
                    pset = preds_m.get(key, [])
                    loss = compute_loss(true_item, pset, metric, base_utility=base_utility)
                    risks.append(loss)
                    sizes.append(len(pset))
                raw_risk = float(np.mean(risks)) if risks else math.nan
                adjusted_risk = raw_risk - (1.0 - base_utility) if not math.isnan(raw_risk) else math.nan
                m_risk[m] = adjusted_risk
                m_size[m] = float(np.mean(sizes)) if sizes else math.nan

            usr_risk, usr_size, active_cnt = [], [], []
            for u in users:
                key = (u, current_step)
                true_item = df_step[df_step["user_idx"] == u]["true_item"].values[0]
                pset = ensemble_at_t.get(key, [])
                loss = compute_loss(true_item, pset, metric, base_utility=base_utility)
                usr_risk.append(loss)
                usr_size.append(len(pset))
                active_cnt.append(sum(1 for m in model_ids if (u, current_step) in prediction_sets[m]))

            ens_risk_raw = float(np.mean(usr_risk))
            ens_risk = ens_risk_raw - (1.0 - base_utility)
            ens_sz = float(np.mean(usr_size))
            avg_active = float(np.mean(active_cnt))

            suffix = f"_{metric}"
            row = {
                "step": current_step,
                "delta": round(delta, 3),
                f"ensemble_risk{suffix}": round(ens_risk, 4),
                f"ensemble_size{suffix}": round(ens_sz, 2),
                f"ensemble_risk_raw{suffix}": round(ens_risk_raw, 4),
                f"avg_active_models{suffix}": round(avg_active, 2)
            }
            for m in model_ids:
                row[f"lambda_{m}{suffix}"] = round(lambdas[current_step][m], 4)
                row[f"risk_{m}{suffix}"] = round(m_risk[m], 4) if not math.isnan(m_risk[m]) else math.nan
                row[f"size_{m}{suffix}"] = round(m_size[m], 2) if not math.isnan(m_size[m]) else math.nan
            diag_rows.append(row)

        skip_updates = (current_step in skip_steps) or (
            frozen_inference and phase_files and phase_file == last_phase_file)

        if not skip_updates:
            weights[current_step+1] = update_weights(set_sizes, current_step, eta)

            updated_lambdas, updated_segments = adaptive_update(
                current_lambdas=lambdas[current_step-1] if current_step > 0 else initial_lambdas,
                loss_traces=losses,
                prediction_sets=prediction_sets,
                df_models=df_models,
                current_step=current_step,
                prev_segment_starts=segment_starts[current_step],
                alpha=alpha,
                eta_t=eta,
                gamma=gamma,
                metric=metric,
                base_utility=base_utility
            )
            lambdas[current_step+1] = updated_lambdas
            segment_starts[current_step+1] = updated_segments
        else:
            weights[current_step+1] = weights[current_step].copy()
            lambdas[current_step+1] = lambdas[current_step].copy()
            segment_starts[current_step+1] = segment_starts[current_step]

        current_step += 1

    return lambdas, weights, ensemble_sets, model_risks, diag_rows
