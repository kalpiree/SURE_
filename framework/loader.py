import os
import pandas as pd
import numpy as np
import ast
import argparse

def parse_list_column(column_value):
    try:
        return ast.literal_eval(column_value)
    except:
        return []

def normalize_scores(scores):
    scores = np.array(scores, dtype=float)
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return [0.5] * len(scores)
    return ((scores - min_val) / (max_val - min_val)).tolist()

def normalize_loss_column(losses):
    losses = np.array(losses, dtype=float)
    min_val = losses.min()
    max_val = losses.max()
    return (losses - min_val) / (max_val - min_val + 1e-8) if max_val > min_val else np.ones_like(losses)

def normalize_model_folder(model_dir: str, output_model_dir: str):
    os.makedirs(output_model_dir, exist_ok=True)

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        input_path = os.path.join(model_dir, fname)
        output_path = os.path.join(output_model_dir, fname)

        df = pd.read_csv(input_path)
        df["candidate_items"] = df["candidate_items"].apply(parse_list_column)
        df["scores"] = df["scores"].apply(parse_list_column)
        df["scores"] = df["scores"].apply(normalize_scores)

        if "loss" in df.columns:
            df["loss"] = normalize_loss_column(df["loss"])

        df.to_csv(output_path, index=False)
        print(f"Normalized: {output_path}")

def normalize_dataset(dataset: str, subdataset: str, output_root: str):
    input_root = os.path.join("datasets", dataset, subdataset)
    output_base = os.path.join(output_root, dataset, subdataset)

    for model_name in os.listdir(input_root):
        model_path = os.path.join(input_root, model_name)
        if os.path.isdir(model_path):
            normalize_model_folder(model_path, os.path.join(output_base, model_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Top-level dataset (e.g. fmlp_runs)")
    parser.add_argument("--subdataset", required=True, help="Subdataset name (e.g. goodreads)")
    parser.add_argument("--output_root", default="datasets_", help="Output folder root (default: datasets_)")
    args = parser.parse_args()

    normalize_dataset(args.dataset, args.subdataset, args.output_root)
    
    