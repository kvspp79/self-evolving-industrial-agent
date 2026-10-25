
"""
STEP 10H — Comparative Safety-Gate Evaluation
NASA C-MAPSS FD001

Gate A: Original rule
  - No validation metric worsens by more than 2%.
  - At least one validation metric improves by 1% or more.

Gate B: NASA-prioritized rule
  - Validation NASA score improves by at least 1%.
  - Validation RMSE and MAE do not worsen by more than 2%.

Both gates are evaluated on identical engine-level splits.
Final held-out engines are not used to choose either gate.

Synthetic shift is used for controlled experimentation only.
This does not establish robustness to real industrial drift.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

DATA_FILE = ROOT / "outputs" / "step01" / "train_FD001_processed.csv"
OUT_DIR = ROOT / "outputs" / "step10H_comparative_safety_gate"

SEEDS = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1111]

ID_COL = "unit_id"
TARGET = "RUL"
CYCLE_COL = "cycle"

SHIFT = {
    "sensor_04": 0.60,
    "sensor_07": -0.60,
    "sensor_11": 0.80,
    "sensor_12": -0.70,
    "sensor_15": 0.65,
}

MAX_WORSENING = 0.02
MIN_IMPROVEMENT = 0.01

N_ESTIMATORS = 250
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = 0.8

# Keep false for standard research runs. Set true only for debugging.
DEBUG = False


# ============================================================
# 2. METRICS
# ============================================================

def nasa_score(y_true, y_pred):
    """
    NASA asymmetric score:
    underprediction is penalized with scale 13;
    overprediction is penalized with scale 10.
    Lower is better.
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    error = np.clip(error, -50, 50)

    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def engine_metrics(frame, predictions):
    """Calculate metrics per engine, then average equally across engines."""
    work = frame[[ID_COL, TARGET]].copy()
    work["prediction"] = np.asarray(predictions, dtype=float)

    records = []

    for engine_id, group in work.groupby(ID_COL, sort=True):
        actual = group[TARGET].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)

        records.append({
            ID_COL: int(engine_id),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "MAE": float(mean_absolute_error(actual, pred)),
            "NASA_score": nasa_score(actual, pred),
        })

    return pd.DataFrame(records)


def summarize_engine_metrics(metrics_df):
    """Equal weight for every engine, regardless of its row count."""
    return {
        "mean_engine_RMSE": float(metrics_df["RMSE"].mean()),
        "mean_engine_MAE": float(metrics_df["MAE"].mean()),
        "mean_engine_NASA_score": float(metrics_df["NASA_score"].mean()),
    }


def improvement_percent(baseline, candidate):
    """
    Positive = candidate improved (lower metric).
    Negative = candidate worsened.
    """
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100.0 * (baseline - candidate) / abs(baseline)


# ============================================================
# 3. DATA AND SYNTHETIC SHIFT
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = {ID_COL, TARGET, CYCLE_COL, *SHIFT.keys()}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="raise").astype(int)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    return df


def apply_synthetic_shift(frame, base_std):
    shifted = frame.copy()

    for sensor, magnitude in SHIFT.items():
        shifted[sensor] = (
            shifted[sensor].astype(float)
            + magnitude * float(base_std[sensor])
        )

    return shifted


# ============================================================
# 4. ENGINE-LEVEL SPLIT
# ============================================================

def split_engines(df, seed):
    engines = np.array(sorted(df[ID_COL].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(engines)

    n = len(engines)
    n_base = int(round(n * 0.55))
    n_adapt = int(round(n * 0.15))
    n_gate = int(round(n * 0.15))

    base_ids = set(engines[:n_base])
    adapt_ids = set(engines[n_base:n_base + n_adapt])
    gate_ids = set(engines[n_base + n_adapt:n_base + n_adapt + n_gate])
    final_ids = set(engines[n_base + n_adapt + n_gate:])

    if set.union(base_ids, adapt_ids, gate_ids, final_ids) != set(engines):
        raise RuntimeError("Engine split does not cover all engines.")

    groups = {
        "base": df[df[ID_COL].isin(base_ids)].copy(),
        "adaptation": df[df[ID_COL].isin(adapt_ids)].copy(),
        "gate": df[df[ID_COL].isin(gate_ids)].copy(),
        "final": df[df[ID_COL].isin(final_ids)].copy(),
    }

    counts = {name: frame[ID_COL].nunique()
              for name, frame in groups.items()}

    if sum(counts.values()) != len(engines):
        raise RuntimeError("Engine split overlap or count mismatch.")

    return groups, counts


# ============================================================
# 5. MODEL TRAINING
# ============================================================

def get_features(df):
    excluded = {ID_COL, TARGET}
    features = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]

    if CYCLE_COL not in features:
        raise ValueError("Cycle must be included as a numeric feature.")

    if not features:
        raise ValueError("No numeric model features found.")

    return features


def build_model(seed):
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=seed,
        n_jobs=-1,
    )


def engine_balanced_weights(base, adaptation):
    """
    Equal total weight for each base engine.
    Adaptation engines receive twice the total weight of base engines.
    Normalize weights to mean 1.
    """
    base_counts = base.groupby(ID_COL)[ID_COL].transform("size")
    adapt_counts = adaptation.groupby(ID_COL)[ID_COL].transform("size")

    base_weights = 1.0 / base_counts.to_numpy(dtype=float)
    adapt_weights = 2.0 / adapt_counts.to_numpy(dtype=float)

    weights = np.concatenate([base_weights, adapt_weights])
    weights /= weights.mean()

    return weights


def fit_models(groups, features, seed):
    base = groups["base"]
    adaptation = apply_synthetic_shift(
        groups["adaptation"],
        base[ list(SHIFT.keys()) ].std(ddof=0),
    )

    base_X = base[features]
    base_y = base[TARGET].to_numpy(dtype=float)

    frozen = build_model(seed)
    frozen.fit(base_X, base_y)

    combined = pd.concat([base, adaptation], ignore_index=True)
    weights = engine_balanced_weights(base, adaptation)

    candidate = build_model(seed)
    candidate.fit(
        combined[features],
        combined[TARGET].to_numpy(dtype=float),
        sample_weight=weights,
    )

    return frozen, candidate, adaptation


# ============================================================
# 6. EVALUATION AND GATE RULES
# ============================================================

def evaluate_model(model, frame, features):
    predictions = model.predict(frame[features])
    per_engine = engine_metrics(frame, predictions)
    return summarize_engine_metrics(per_engine), per_engine


def gate_comparison(baseline_metrics, candidate_metrics):
    changes = {}

    for metric in [
        "mean_engine_RMSE",
        "mean_engine_MAE",
        "mean_engine_NASA_score",
    ]:
        changes[metric] = improvement_percent(
            baseline_metrics[metric],
            candidate_metrics[metric],
        )

    return changes


def original_gate(changes):
    """
    Original rule:
    every metric must be no worse than -2%;
    at least one metric must improve by >=1%.
    """
    values = np.asarray(list(changes.values()), dtype=float)

    all_within_tolerance = (
        np.isfinite(values).all()
        and np.all(values >= -100.0 * MAX_WORSENING)
    )

    at_least_one_improves = (
        np.isfinite(values).any()
        and np.any(values >= 100.0 * MIN_IMPROVEMENT)
    )

    return bool(all_within_tolerance and at_least_one_improves)


def nasa_prioritized_gate(changes):
    """
    Proposed rule:
    NASA score must improve by >=1%;
    RMSE and MAE must each avoid worsening by >2%.
    """
    rmse_ok = (
        np.isfinite(changes["mean_engine_RMSE"])
        and changes["mean_engine_RMSE"] >= -100.0 * MAX_WORSENING
    )
    mae_ok = (
        np.isfinite(changes["mean_engine_MAE"])
        and changes["mean_engine_MAE"] >= -100.0 * MAX_WORSENING
    )
    nasa_ok = (
        np.isfinite(changes["mean_engine_NASA_score"])
        and changes["mean_engine_NASA_score"]
        >= 100.0 * MIN_IMPROVEMENT
    )

    return bool(rmse_ok and mae_ok and nasa_ok)


# ============================================================
# 7. ONE SEED RUN
# ============================================================

def run_seed(df, seed, features):
    groups, counts = split_engines(df, seed)

    print("\n" + "=" * 68)
    print(f"SEED {seed}")
    print("Engine counts:", counts)

    base_std = groups["base"][list(SHIFT.keys())].std(ddof=0)

    frozen, candidate, shifted_adaptation = fit_models(
        groups, features, seed
    )

    shifted_gate = apply_synthetic_shift(groups["gate"], base_std)
    shifted_final = apply_synthetic_shift(groups["final"], base_std)

    frozen_gate_metrics, frozen_gate_engines = evaluate_model(
        frozen, shifted_gate, features
    )
    candidate_gate_metrics, candidate_gate_engines = evaluate_model(
        candidate, shifted_gate, features
    )

    validation_changes = gate_comparison(
        frozen_gate_metrics, candidate_gate_metrics
    )

    accepted_a = original_gate(validation_changes)
    accepted_b = nasa_prioritized_gate(validation_changes)

    print("Gate validation improvements (%):")
    for metric, value in validation_changes.items():
        print(f"  {metric}: {value:.3f}%")

    print("Original gate:", "ACCEPT" if accepted_a else "REJECT")
    print("NASA-prioritized gate:", "ACCEPT" if accepted_b else "REJECT")

    # Final held-out evaluation occurs after both rules are determined.
    frozen_final_metrics, frozen_final_engines = evaluate_model(
        frozen, shifted_final, features
    )
    candidate_final_metrics, candidate_final_engines = evaluate_model(
        candidate, shifted_final, features
    )

    # Each gate either keeps frozen model or accepts the same candidate.
    chosen_a = candidate_final_metrics if accepted_a else frozen_final_metrics
    chosen_b = candidate_final_metrics if accepted_b else frozen_final_metrics

    print("Final held-out engine counts:", counts["final"])
    print("Frozen final metrics:", frozen_final_metrics)
    print("Candidate final metrics:", candidate_final_metrics)

    model_rows = []
    per_engine_rows = []

    for model_name, metrics, per_engine in [
        ("frozen_baseline", frozen_final_metrics, frozen_final_engines),
        ("unconditional_candidate", candidate_final_metrics,
         candidate_final_engines),
        ("original_gate_model", chosen_a,
         candidate_final_engines if accepted_a else frozen_final_engines),
        ("nasa_prioritized_gate_model", chosen_b,
         candidate_final_engines if accepted_b else frozen_final_engines),
    ]:
        model_rows.append({
            "seed": seed,
            "model": model_name,
            "n_final_engines": counts["final"],
            **metrics,
        })

        for _, row in per_engine.iterrows():
            per_engine_rows.append({
                "seed": seed,
                "model": model_name,
                **row.to_dict(),
            })

    gate_row = {
        "seed": seed,
        **{f"n_{name}_engines": count for name, count in counts.items()},
        **{f"validation_improvement_pct_{k}": v
           for k, v in validation_changes.items()},
        "original_gate_accepted": accepted_a,
        "nasa_prioritized_gate_accepted": accepted_b,
    }

    # Retain engine-level validation metrics for traceability.
    validation_rows = []
    for model_name, per_engine in [
        ("frozen_baseline", frozen_gate_engines),
        ("candidate", candidate_gate_engines),
    ]:
        for _, row in per_engine.iterrows():
            validation_rows.append({
                "seed": seed,
                "model": model_name,
                **row.to_dict(),
            })

    return model_rows, per_engine_rows, gate_row, validation_rows


# ============================================================
# 8. MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    features = get_features(df)

    print("=" * 68)
    print("STEP 10H — COMPARATIVE SAFETY-GATE EVALUATION")
    print("=" * 68)
    print("Dataset:", DATA_FILE)
    print("Rows:", len(df))
    print("Engines:", df[ID_COL].nunique())
    print("Numeric model features:", len(features))
    print("Seeds:", SEEDS)
    print("Synthetic shift:", SHIFT)
    print("Gate A: original rule")
    print("Gate B: NASA-prioritized rule")

    if DEBUG:
        seeds = SEEDS[:2]
    else:
        seeds = SEEDS

    all_model_rows = []
    all_engine_rows = []
    all_gate_rows = []
    all_validation_rows = []

    for seed in seeds:
        model_rows, engine_rows, gate_row, validation_rows = run_seed(
            df, seed, features
        )
        all_model_rows.extend(model_rows)
        all_engine_rows.extend(engine_rows)
        all_gate_rows.append(gate_row)
        all_validation_rows.extend(validation_rows)

    models_df = pd.DataFrame(all_model_rows)
    engines_df = pd.DataFrame(all_engine_rows)
    gates_df = pd.DataFrame(all_gate_rows)
    validation_df = pd.DataFrame(all_validation_rows)

    models_df.to_csv(
        OUT_DIR / "step10H_final_metrics_by_seed.csv", index=False
    )
    engines_df.to_csv(
        OUT_DIR / "step10H_final_engine_metrics.csv", index=False
    )
    gates_df.to_csv(
        OUT_DIR / "step10H_gate_decisions.csv", index=False
    )
    validation_df.to_csv(
        OUT_DIR / "step10H_gate_validation_engine_metrics.csv", index=False
    )

    # Aggregate across seeds. These are descriptive summaries, not
    # independent-dataset estimates.
    summary_rows = []

    for model_name, group in models_df.groupby("model"):
        row = {
            "model": model_name,
            "successful_seed_runs": int(group["seed"].nunique()),
        }

        for metric in [
            "mean_engine_RMSE",
            "mean_engine_MAE",
            "mean_engine_NASA_score",
        ]:
            row[f"{metric}_mean_across_seeds"] = float(group[metric].mean())
            row[f"{metric}_std_across_seeds"] = float(group[metric].std(ddof=1))
            row[f"{metric}_median_across_seeds"] = float(group[metric].median())

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUT_DIR / "step10H_across_seed_summary.csv", index=False
    )

    gate_summary = {
        "seed_runs": int(len(gates_df)),
        "original_gate_accepted": int(
            gates_df["original_gate_accepted"].sum()
        ),
        "nasa_prioritized_gate_accepted": int(
            gates_df["nasa_prioritized_gate_accepted"].sum()
        ),
        "original_gate_acceptance_rate_percent": float(
            100 * gates_df["original_gate_accepted"].mean()
        ),
        "nasa_prioritized_gate_acceptance_rate_percent": float(
            100 * gates_df["nasa_prioritized_gate_accepted"].mean()
        ),
        "both_accepted": int(
            (
                gates_df["original_gate_accepted"]
                & gates_df["nasa_prioritized_gate_accepted"]
            ).sum()
        ),
        "original_only_accepted": int(
            (
                gates_df["original_gate_accepted"]
                & ~gates_df["nasa_prioritized_gate_accepted"]
            ).sum()
        ),
        "nasa_prioritized_only_accepted": int(
            (
                ~gates_df["original_gate_accepted"]
                & gates_df["nasa_prioritized_gate_accepted"]
            ).sum()
        ),
        "neither_accepted": int(
            (
                ~gates_df["original_gate_accepted"]
                & ~gates_df["nasa_prioritized_gate_accepted"]
            ).sum()
        ),
    }

    report = {
        "step": "10H",
        "title": "Comparative Safety-Gate Evaluation",
        "dataset": str(DATA_FILE),
        "output_directory": str(OUT_DIR),
        "seeds": seeds,
        "feature_columns": features,
        "synthetic_shift": SHIFT,
        "gate_definitions": {
            "original_gate": (
                "No validation metric worsens by more than 2%; "
                "at least one improves by 1% or more."
            ),
            "nasa_prioritized_gate": (
                "Validation NASA score improves by at least 1%; "
                "RMSE and MAE do not worsen by more than 2%."
            ),
        },
        "gate_summary": gate_summary,
        "across_seed_summary": summary_df.to_dict(orient="records"),
        "limitations": [
            "Sensor shift is synthetic, not observed industrial drift.",
            "Seeds reuse the same source dataset.",
            "The final held-out engines are used only for evaluation.",
            "The two gate rules are predefined before examining this run's final results.",
            "The proposed gate is a hypothesis, not assumed to be superior.",
            "A gate that performs well on these seeds may not generalize to other fleets.",
            "Per-engine NASA score sums depend on the number of rows per engine.",
            "No causal interpretation is justified by this experiment.",
        ],
        "output_files": {
            "final_metrics_by_seed": "step10H_final_metrics_by_seed.csv",
            "final_engine_metrics": "step10H_final_engine_metrics.csv",
            "gate_decisions": "step10H_gate_decisions.csv",
            "gate_validation_engine_metrics":
                "step10H_gate_validation_engine_metrics.csv",
            "across_seed_summary": "step10H_across_seed_summary.csv",
        },
    }

    with open(
        OUT_DIR / "step10H_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 68)
    print("STEP 10H COMPLETE")
    print("=" * 68)

    print("\nAcross-seed summary:")
    print(summary_df.to_string(index=False))

    print("\nGate summary:")
    print(json.dumps(gate_summary, indent=2))

    print("\nOutput directory:", OUT_DIR)
    print("Report:", OUT_DIR / "step10H_report.json")


if __name__ == "__main__":
    main()
