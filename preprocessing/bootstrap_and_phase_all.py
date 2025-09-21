import os
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

INPUT_DIR = Path("./output")
BASE_OUTPUT_DIR = Path("./processed_datasets")
DATASETS = ["bookcrossing.csv", "gowalla.csv", "lastfm.csv", "steam.csv", "taobao.csv"]
N_MODELS = 10
MIN_INTERACTIONS = 100
NUM_PHASES = 5
EVAL_POINTS = 10
NEG_SAMPLES = 50
RNG_SEED = 42
LOG_LEVEL = "INFO"

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

def densify_ids(df: pd.DataFrame) -> pd.DataFrame:
    df["user_idx"], _ = pd.factorize(df["user_idx"])
    df["item_idx"], _ = pd.factorize(df["item_idx"])
    return df

def chunkify(seq: list, n_chunks: int) -> list[list]:
    q = len(seq) // n_chunks
    return [seq[i * q:(i + 1) * q] for i in range(n_chunks)]

def sample_negatives(pool: np.ndarray, k: int, true_item: int) -> list[int]:
    pool = pool[pool != true_item]
    if len(pool) == 0:
        return []
    k = min(k, len(pool))
    return random.sample(list(pool), k)

def build_global_phases(df: pd.DataFrame, dataset_name: str) -> dict:
    logging.info(f"Building global 5-phase split for {dataset_name!r}…")

    phase_data = {p: {"train_lines": [], "eval_rows": []} for p in range(NUM_PHASES)}
    all_items = df["item_idx"].unique()

    user_groups = df.groupby("user_idx")
    skipped_users = 0

    for u, g in tqdm(user_groups, desc="users", unit_scale=True):
        items = g.sort_values("timestamp")["item_idx"].to_numpy().tolist()
        if len(items) < MIN_INTERACTIONS:
            skipped_users += 1
            continue

        chunks = chunkify(items, NUM_PHASES)
        if any(len(c) == 0 for c in chunks):
            skipped_users += 1
            continue

        for phase in range(NUM_PHASES):
            history_so_far = []
            for p in range(phase + 1):
                chunk_p = chunks[p]
                train_part = (chunk_p[:-EVAL_POINTS] if len(chunk_p) > EVAL_POINTS else chunk_p)
                history_so_far.extend(train_part)

            for it in history_so_far:
                phase_data[phase]["train_lines"].append(f"{u} {it}\n")

            true_items = (chunks[phase][-EVAL_POINTS:] if len(chunks[phase]) > EVAL_POINTS else [])

            for t in true_items:
                negs = sample_negatives(pool=all_items, k=NEG_SAMPLES, true_item=t)
                candidates = negs + [t]
                random.shuffle(candidates)

                assert t in candidates

                phase_data[phase]["eval_rows"].append({
                    "user_idx": int(u),
                    "history": [int(x) for x in history_so_far],
                    "true_item": int(t),
                    "candidate_items": [int(x) for x in candidates]
                })

    logging.info(f"Skipped {skipped_users:,} users due to not enough interactions.")

    global_dir = BASE_OUTPUT_DIR / dataset_name / "global_phases"
    global_dir.mkdir(parents=True, exist_ok=True)

    for phase in range(NUM_PHASES):
        train_path = global_dir / f"train_phase{phase}.txt"
        with open(train_path, "w") as f:
            f.writelines(phase_data[phase]["train_lines"])

        eval_path = global_dir / f"eval_phase{phase}.csv"
        df_eval = pd.DataFrame(phase_data[phase]["eval_rows"])
        df_eval.to_csv(eval_path, index=False)

        if not df_eval.empty:
            cand_lengths = df_eval["candidate_items"].apply(lambda x: len(eval(str(x))))

        logging.info(
            f"Phase {phase}:  {len(phase_data[phase]['train_lines']):,} train interactions, "
            f"{len(phase_data[phase]['eval_rows']):,} eval rows"
        )

    return phase_data

def build_bootstrapped_models(df: pd.DataFrame, dataset_name: str, phase_data: dict):
    users = df["user_idx"].unique()

    for i in range(N_MODELS):
        model_name = f"model_{i}"
        model_dir = BASE_OUTPUT_DIR / dataset_name / "phased_data" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        sampled_users = np.random.choice(users, size=len(users), replace=True)
        unique_users = np.unique(sampled_users)
        user_mask = set(unique_users)

        for phase in range(NUM_PHASES):
            train_lines = [ln for ln in phase_data[phase]["train_lines"]
                           if int(ln.split()[0]) in user_mask]
            with open(model_dir / f"train_phase{phase}.txt", "w") as f:
                f.writelines(train_lines)

            eval_rows = [row for row in phase_data[phase]["eval_rows"]
                         if row["user_idx"] in user_mask]
            pd.DataFrame(eval_rows).to_csv(
                model_dir / f"eval_phase{phase}.csv",
                index=False
            )

def process_dataset(csv_file: str):
    dataset_name = csv_file.replace("_final.csv", "")
    logging.info(f"\n=========  DATASET: {dataset_name}  =========")

    df = pd.read_csv(INPUT_DIR / csv_file)
    df = densify_ids(df)

    phase_data = build_global_phases(df, dataset_name)
    build_bootstrapped_models(df, dataset_name, phase_data)

if __name__ == "__main__":
    for file in DATASETS:
        process_dataset(file)

    logging.info("All done!")
