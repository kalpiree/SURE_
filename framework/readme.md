# SURE: Shift-aware, User-adaptive, Risk-controlled rEcommendations

SURE (Shift-aware, User-adaptive, Risk-controlled rEcommendations) is a framework for dynamically calibrating and aggregating predictions from multiple sequential recommender models, under user-level uncertainty constraints.

## 📁 Project Structure

```
.
├── calibration/         # Calibration modules (risk, shift, adaptive update)
├── data/                # Data loader and normalizer
├── outputs/             # Generated results
├── datasets/            # Raw evaluation files (input)
├── datasets_/           # Normalized evaluation files (output)
├── sure.sh              # Shell script to run DUAR
├── main_.py             # Main driver to run DAUR
├── run_daur_.py         # Core DAUR logic
```

##  Setup

Requires:

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `torch`

Install dependencies (if needed):

```bash
pip install pandas numpy scikit-learn torch
```

##  How to Run

### Step 1: Normalize Evaluation Files

This step converts raw CSVs into a format DUAR expects:

```bash
python data/loader.py --dataset sasrec --subdataset goodreads --output_root datasets_
```

### Step 2: Run DUAR

Use the provided shell script:

```bash
bash duar.sh
```

Or run manually:

```bash
python main_.py \
  --dataset sasrec \
  --subdataset goodreads \
  --alphas 0.1 \
  --etas 0.5 \
  --gamma 2.0 \
  --base_utilities recall=0.67 \
  --output_dir outputs \
  --freeze_inference \
  --max_pred_set_size 50
```

### Arguments

| Argument              | Description |
|-----------------------|-------------|
| `--dataset`           | Dataset name (e.g., `sasrec`) |
| `--subdataset`        | Subdataset (e.g., `goodreads`) |
| `--alphas`            | Risk level (e.g., 0.1) |
| `--etas`              | Learning rate for lambda |
| `--gamma`             | Weight decay over segments |
| `--base_utilities`    | Minimum expected utility (e.g., `recall=0.67`) |
| `--max_pred_set_size` | Max items in prediction set |
| `--freeze_inference`  | Freeze updates during final phase |

## Output

Results are saved under:

```
outputs/sasrec/goodreads/alpha_010/
└── detailed_snapshots.csv
```

It contains:

- Ensemble risk and size over time
- Model-specific lambda values and risk



