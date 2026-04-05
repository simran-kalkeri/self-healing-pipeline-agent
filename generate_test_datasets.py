"""
generate_test_datasets.py
=========================
Creates 7 CSV files — one clean baseline and one for each failure mode
supported by the Self-Healing ML Pipeline dashboard.

Each file replicates the *exact* logic from pipeline.py so the agent sees
precisely the same data distributions it was designed to heal.

Usage:
    python generate_test_datasets.py

Output directory: ./test_datasets/
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match pipeline.py constants)
# ─────────────────────────────────────────────────────────────────────────────

N              = 500
SEED_BASE      = 42
TARGET_COLUMN  = "target"
OUT_DIR        = Path("test_datasets")

# ─────────────────────────────────────────────────────────────────────────────
# BASE DATA GENERATOR  (identical to pipeline.generate_synthetic_data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_base(n: int = N, seed: int = SEED_BASE) -> pd.DataFrame:
    """
    Logistic-rule dataset where income (60%) and age (40%) predict target.
    Clean F1 typically ~0.87 with LogisticRegression → passes 0.80 goal.
    """
    rng    = np.random.default_rng(seed)
    age    = rng.integers(18, 70, size=n).astype(float)
    income = rng.integers(20_000, 120_000, size=n).astype(float)
    gender = rng.choice(["M", "F"], size=n)

    age_norm    = (age - 18) / (70 - 18)
    income_norm = (income - 20_000) / 100_000
    score       = 0.6 * income_norm + 0.4 * age_norm
    noise       = rng.normal(0, 0.15, size=n)
    prob        = 1 / (1 + np.exp(-(score - 0.5 + noise) * 6))
    target      = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "age":         age,
        "gender":      gender,
        "income":      income,
        TARGET_COLUMN: target,
    })


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE INJECTORS  (identical to pipeline.inject_failures)
# ─────────────────────────────────────────────────────────────────────────────

def inject_missing_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the 'income' column entirely.
    Expected agent fix: ADD_MISSING_COLUMNS
    Detector trigger: MISSING_COLUMNS:income
    """
    return df.drop(columns=["income"], errors="ignore")


def inject_type_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces ~30% of 'age' values with the string 'N/A'.
    Expected agent fix: CAST_DATATYPE
    Detector trigger: TYPE_MISMATCH:age
    """
    rng = np.random.default_rng(99)
    df = df.copy()
    df["age"] = df["age"].astype(str).apply(
        lambda x: "N/A" if rng.random() < 0.3 else x
    )
    return df


def inject_low_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Severe class imbalance — only 10% minority class (50 rows out of 500).
    Features are shuffled so they no longer predict target.
    Expected agent fix: REBALANCE_DATA (SMOTE)
    Detector trigger: LOW_PERFORMANCE:f1=<low>
    """
    rng = np.random.default_rng(99)
    n = len(df)
    df = df.copy()
    n_minority = max(int(n * 0.10), 5)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_minority, replace=False)] = 1
    df[TARGET_COLUMN] = labels
    return df


def inject_compound(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two combined failures:
      1. 'income' column missing
      2. Only 5% class-1 rows remain (extreme imbalance)
    Expected agent fixes (in sequence): ADD_MISSING_COLUMNS → REBALANCE_DATA
    Detector trigger: MISSING_COLUMNS:income first, then class imbalance
    """
    rng = np.random.default_rng(99)
    df = df.drop(columns=["income"], errors="ignore").copy()

    class1_idx = df.index[df[TARGET_COLUMN] == 1].tolist()
    n_keep     = max(int(len(df) * 0.05), 5)
    top_class1 = df.loc[class1_idx, "age"].sort_values(ascending=False).index[:n_keep]
    flip_mask  = df.index.isin(class1_idx) & ~df.index.isin(top_class1)
    df.loc[flip_mask, TARGET_COLUMN] = 0
    return df


def inject_near_goal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mild label noise: ~10% of labels randomly flipped.
    Brings F1 from ~0.87 → ~0.75–0.77 (near-goal band 0.74–0.80).
    Expected agent fix: TUNE_HYPERPARAMETERS (precision tuning)
    Detector trigger: LOW_PERFORMANCE:f1=~0.76  +  near_goal=True
    """
    rng  = np.random.default_rng(7)
    df   = df.copy()
    mask = rng.random(len(df)) < 0.10
    df.loc[mask, TARGET_COLUMN] = 1 - df.loc[mask, TARGET_COLUMN]
    return df


def inject_partial_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """
    30% of 'income' values silently replaced with NaN.
    Column exists but is heavily null-corrupted.
    Expected agent fix: CAST_DATATYPE (fills NaN with median)
    Detector trigger: NULL_VALUES:income:30%
    """
    rng  = np.random.default_rng(99)
    df   = df.copy()
    mask = rng.random(len(df)) < 0.30
    df.loc[mask, "income"] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "filename":    "clean_baseline.csv",
        "failure":     "none",
        "inject_fn":   None,
        "description": "Clean data, no failures. Agent should pass in attempt #1 (F1≈0.87).",
        "expected_fix": "STOP (goal reached immediately)",
    },
    {
        "filename":    "missing_column.csv",
        "failure":     "MISSING_COLUMN",
        "inject_fn":   inject_missing_column,
        "description": "'income' column is completely absent.",
        "expected_fix": "ADD_MISSING_COLUMNS",
    },
    {
        "filename":    "type_mismatch.csv",
        "failure":     "TYPE_MISMATCH",
        "inject_fn":   inject_type_mismatch,
        "description": "~30% of 'age' values are the string 'N/A' (non-numeric).",
        "expected_fix": "CAST_DATATYPE",
    },
    {
        "filename":    "low_performance.csv",
        "failure":     "LOW_PERFORMANCE",
        "inject_fn":   inject_low_performance,
        "description": "Extreme class imbalance — only 10% minority class (≈50 rows).",
        "expected_fix": "REBALANCE_DATA (SMOTE)",
    },
    {
        "filename":    "compound.csv",
        "failure":     "COMPOUND",
        "inject_fn":   inject_compound,
        "description": "Two failures: 'income' missing + only 5% class-1 rows.",
        "expected_fix": "ADD_MISSING_COLUMNS then REBALANCE_DATA",
    },
    {
        "filename":    "near_goal.csv",
        "failure":     "NEAR_GOAL",
        "inject_fn":   inject_near_goal,
        "description": "~10% of labels randomly flipped. F1 drops into near-goal zone (0.74–0.80).",
        "expected_fix": "TUNE_HYPERPARAMETERS",
    },
    {
        "filename":    "partial_corruption.csv",
        "failure":     "PARTIAL_CORRUPTION",
        "inject_fn":   inject_partial_corruption,
        "description": "30% of 'income' values are NaN (column exists but corrupted).",
        "expected_fix": "CAST_DATATYPE (median fill)",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    base_df = generate_base()

    print("=" * 62)
    print("  Self-Healing ML Pipeline — Test Dataset Generator")
    print(f"  Output directory: ./{OUT_DIR}/")
    print("=" * 62)

    for spec in DATASETS:
        df = base_df.copy()

        if spec["inject_fn"] is not None:
            df = spec["inject_fn"](df)

        path = OUT_DIR / spec["filename"]
        df.to_csv(path, index=False)

        print(f"\n✅  {spec['filename']}")
        print(f"    Failure   : {spec['failure']}")
        print(f"    Shape     : {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"    Columns   : {list(df.columns)}")
        if TARGET_COLUMN in df.columns:
            dist = df[TARGET_COLUMN].value_counts().to_dict()
            print(f"    Target    : {dist}")
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"    Nulls     : {null_counts[null_counts > 0].to_dict()}")
        print(f"    Expected  : {spec['expected_fix']}")
        print(f"    Note      : {spec['description']}")

    print("\n" + "=" * 62)
    print(f"  {len(DATASETS)} CSV files written to ./{OUT_DIR}/")
    print()
    print("  HOW TO USE IN THE DASHBOARD:")
    print("  1. Open http://localhost:8501")
    print("  2. Upload any CSV via the file uploader")
    print("  3. Select the MATCHING failure mode in the dropdown")
    print("     (the dashboard uses the dropdown to call run_pipeline,")
    print("      but the data health panel will show the CSV's diagnostics)")
    print()
    print("  ⚠  IMPORTANT — Dashboard upload behaviour:")
    print("     The pipeline always generates its own synthetic data")
    print("     internally (run_pipeline ignores the uploaded CSV for")
    print("     actual training). The upload is shown in the Data Health")
    print("     panel so you can VISUALLY inspect the failure pattern.")
    print("     To test a specific failure end-to-end, use the DROPDOWN.")
    print("=" * 62)


if __name__ == "__main__":
    main()
