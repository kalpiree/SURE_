import os
import argparse
import pandas as pd
import numpy as np
from data.loader import parse_list_column
from calibration.lambda_search_ import search_valid_lambda_step0
from run_daur_ import run_daur
from calibration.risk_estimator import compute_utility

def main():
    parser = argparse.ArgumentParser(description="Run DAUR automation over alpha/delta sweep")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subdataset", type=str, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    parser.add_argument("--etas", type=float, nargs="+", default=[0.5])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--freeze_inference", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_pred_set_size", type=int, default=None)
    parser.add_argument(
        "--base_utilities",
        type=str,
        required=True,
        help="Comma-separated per-metric base utilities, e.g., recall=0.67,ndcg=0.6,mrr=0.4"
    )
    args = parser.parse_args()

    import inspect
    valid_metrics = set()
    source = inspect.getsource(compute_utility)
    for m in ["recall", "ndcg", "mrr"]:
        if f'if metric == "{m}"' in source or f"elif metric == \"{m}\"" in source:
            valid_metrics.add(m)

    base_util_map = {}
    for pair in args.base_utilities.split(","):
        k, v = pair.split("=")
        k = k.strip()
        if k not in valid_metrics:
            raise ValueError(f"Unsupported metric '{k}'. Supported: {sorted(valid_metrics)}")
        base_util_map[k] = float(v)

    metrics = list(base_util_map.keys())
    delta_values = [0.05]

    for alpha in args.alphas:
        output_base = os.path.join(
            args.output_dir, args.dataset, args.subdataset, f"alpha_{int(alpha * 100):03d}"
        )
        os.makedirs(output_base, exist_ok=True)

        base_dir = os.path.join("datasets_", args.dataset, args.subdataset)
        model_folders = sorted([
            f for f in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, f))
        ])
        df_models = {}
        model_ids = list(range(len(model_folders)))

        for model_id, model_folder in enumerate(model_folders):
            full_path = os.path.join(base_dir, model_folder)
            df_all = []
            for phase_idx in range(5):
                phase_file = os.path.join(full_path, f"phase{phase_idx}_eval_output.csv")
                df_phase = pd.read_csv(phase_file)
                df_phase["step"] = df_phase["step"] + phase_idx * 10
                df_all.append(df_phase)
            df_model = pd.concat(df_all).reset_index(drop=True)
            df_model["candidate_items"] = df_model["candidate_items"].apply(parse_list_column)
            df_model["scores"] = df_model["scores"].apply(parse_list_column)
            df_model["normalized_scores"] = df_model["scores"]
            df_models[model_id] = df_model

        combined_snapshots = []

        for metric in metrics:
            metric_snapshots = []

            for delta in delta_values:
                initial_lambdas = {}
                for model_id, df_model in df_models.items():
                    lambda_0, _, _ = search_valid_lambda_step0(
                        df_model,
                        alpha=alpha,
                        metric=metric,
                        step_t=0,
                        max_pred_set_size=args.max_pred_set_size,
                        base_utility=base_util_map[metric],
                        delta=delta
                    )
                    lambda_0 = lambda_0 if lambda_0 is not None else 1.0
                    initial_lambdas[model_id] = lambda_0

                initial_weights = {model_id: 1.0 / len(model_ids) for model_id in model_ids}

                _, _, _, _, diag_rows = run_daur(
                    phase_files=[],
                    model_ids=model_ids,
                    initial_lambdas=initial_lambdas,
                    initial_weights=initial_weights,
                    df_models=df_models,
                    alpha=alpha,
                    eta=args.etas[0],
                    gamma=args.gamma,
                    metric=metric,
                    save_outputs=False,
                    output_dir=output_base,
                    frozen_inference=args.freeze_inference,
                    max_pred_set_size=args.max_pred_set_size,
                    base_utility=base_util_map[metric],
                    delta=delta
                )

                df_diag = pd.DataFrame(diag_rows)
                df_diag = df_diag.add_suffix(f"_{metric}")
                df_diag.rename(columns={f"step_{metric}": "step", f"delta_{metric}": "delta"}, inplace=True)
                metric_snapshots.append(df_diag)

            df_metric_full = pd.concat(metric_snapshots).sort_values(by=["step", "delta"])
            if combined_snapshots:
                combined_snapshots[0] = pd.merge(
                    combined_snapshots[0], df_metric_full, on=["step", "delta"], how="outer"
                )
            else:
                combined_snapshots.append(df_metric_full)

        final_df = combined_snapshots[0].sort_values(by=["step", "delta"])
        final_df.to_csv(os.path.join(output_base, "detailed_snapshots.csv"), index=False)

if __name__ == "__main__":
    main()
