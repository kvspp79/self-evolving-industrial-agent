
"""
STEP 10E — SELF-EVOLUTION DIAGNOSIS AND CANDIDATE EVALUATION
Dataset: NASA C-MAPSS FD001
Python: 3.13+

Purpose:
1. Diagnose why previous updates failed the safety gate.
2. Compare predefined replay-learning strategies.
3. Weight newly labelled adaptation data differently.
4. Evaluate validation metrics at the engine level.
5. Accept only candidates that satisfy predefined criteria.
6. Evaluate selected candidates on untouched held-out engines.

IMPORTANT:
- The sensor shift is synthetic.
- The official NASA test file is not used.
- Engines are split before rows are selected.
- Final evaluation engines are not used for model selection.
- Acceptance is not guaranteed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. FIXED EXPERIMENT CONFIGURATION
# ============================================================

SEED = 20261025
SHIFT_SEED = 4107

BASE_FRACTION = 0.55
ADAPT_FRACTION = 0.15
GATE_FRACTION = 0.15

N_TREES = 100
MIN_SAMPLES_LEAF = 2
MAX_DEPTH = 18

MAX_ALLOWED_DEGRADATION = 0.02
MIN_REQUIRED_IMPROVEMENT = 0.01

TARGET = "RUL"
GROUP = "unit_id"

NON_FEATURES = {TARGET, GROUP}
REQUIRED_COLUMNS = {GROUP, "cycle", TARGET}

# Predefined synthetic shift in training-standard-deviation units.
SHIFT_SPEC = {
    "sensor_04": +0.60,
    "sensor_07": -0.60,
    "sensor_11": +0.80,
    "sensor_12": -0.70,
    "sensor_15": +0.65,
}

MONITORED_METRICS = [
    "mean_engine_RMSE",
    "mean_engine_MAE",
    "mean_engine_NASA_score",
]


# ============================================================
# 2. LOCATE PROJECT AND INPUT DATA
# ============================================================

def find_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1],
        Path.cwd(),
    ]

    for candidate in candidates:
        expected = (
            candidate
            / "outputs"
            / "step01"
            / "train_FD001_processed.csv"
        )

        if expected.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find "
        "outputs\\step01\\train_FD001_processed.csv. "
        "Save this script in the NASA project's src folder."
    )


# ============================================================
# 3. MODEL FACTORY
# ============================================================

def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=MAX_DEPTH,
        random_state=SEED,
        n_jobs=-1,
    )


# ============================================================
# 4. SYNTHETIC SENSOR SHIFT
# ============================================================

def apply_shift(
    frame: pd.DataFrame,
    features: list[str],
    base_stds: pd.Series,
) -> pd.DataFrame:

    result = frame.copy()

    for sensor, shift_units in SHIFT_SPEC.items():
        if sensor not in features:
            raise ValueError(
                f"Shift feature not found: {sensor}"
            )

        scale = float(base_stds[sensor])

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Invalid training scale for {sensor}"
            )

        result[sensor] = (
            result[sensor] + shift_units * scale
        )

    return result


# ============================================================
# 5. NASA PHM SCORE
# ============================================================

def nasa_score(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    error = (
        np.asarray(predicted, dtype=float)
        - np.asarray(actual, dtype=float)
    )

    # Clip exponent inputs only to avoid numerical overflow.
    # This does not affect ordinary errors in the normal range.
    exponent = np.where(
        error < 0,
        np.clip(-error / 13.0, -50, 50),
        np.clip(error / 10.0, -50, 50),
    )

    penalties = np.exp(exponent) - 1.0

    return float(np.sum(penalties))


# ============================================================
# 6. PER-ENGINE EVALUATION
# ============================================================

def per_engine_metrics(
    model: RandomForestRegressor,
    frame: pd.DataFrame,
    features: list[str],
    label: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Calculate metrics separately for each engine.

    The reported aggregate metrics are means of engine-level
    metrics, so long engines do not dominate the aggregate.
    """

    predictions = model.predict(frame[features])

    scored = frame[
        [GROUP, "cycle", TARGET]
    ].copy()

    scored["prediction"] = predictions
    scored["error"] = (
        scored["prediction"] - scored[TARGET]
    )

    records = []

    for engine_id, group in scored.groupby(
        GROUP,
        sort=False,
    ):
        actual = group[TARGET].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)

        records.append({
            GROUP: engine_id,
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(actual, predicted)
                )
            ),
            "MAE": float(
                mean_absolute_error(actual, predicted)
            ),
            "NASA_score": nasa_score(actual, predicted),
            "mean_error": float(
                np.mean(predicted - actual)
            ),
        })

    engine_df = pd.DataFrame(records)

    aggregate = {
        "evaluation_slice": label,
        "n_engines": int(len(engine_df)),
        "mean_engine_RMSE": float(
            engine_df["RMSE"].mean()
        ),
        "mean_engine_MAE": float(
            engine_df["MAE"].mean()
        ),
        "mean_engine_NASA_score": float(
            engine_df["NASA_score"].mean()
        ),
        "mean_engine_error": float(
            engine_df["mean_error"].mean()
        ),
    }

    return aggregate, engine_df


# ============================================================
# 7. SAFETY-GATE EVALUATION
# ============================================================

def gate_candidate(
    incumbent_metrics: dict,
    candidate_metrics: dict,
) -> tuple[bool, str]:
    """
    Acceptance criteria are fixed before final evaluation.

    A candidate is rejected if any monitored mean engine-level
    metric worsens by more than 2%.

    It must also improve at least one monitored metric by 1%
    or more.

    These are experimental thresholds, not universal safety rules.
    """

    failures = []
    improvements = []

    for metric in MONITORED_METRICS:
        old = float(incumbent_metrics[metric])
        new = float(candidate_metrics[metric])

        if not np.isfinite(old) or not np.isfinite(new):
            failures.append(
                f"{metric} is non-finite"
            )
            continue

        allowed = old * (
            1.0 + MAX_ALLOWED_DEGRADATION
        )

        if new > allowed:
            failures.append(
                f"{metric} worsened beyond tolerance "
                f"({old:.5f} -> {new:.5f})"
            )

        if old > 0:
            relative_improvement = (
                old - new
            ) / old

            if (
                relative_improvement
                >= MIN_REQUIRED_IMPROVEMENT
            ):
                improvements.append(
                    f"{metric} improved "
                    f"{relative_improvement:.2%}"
                )

    if failures:
        return False, "Rejected: " + "; ".join(failures)

    if not improvements:
        return (
            False,
            "Rejected: no monitored metric improved by at "
            f"least {MIN_REQUIRED_IMPROVEMENT:.1%}",
        )

    return True, "Accepted: " + "; ".join(improvements)


# ============================================================
# 8. CANDIDATE TRAINING STRATEGIES
# ============================================================

def train_candidate(
    strategy: str,
    base: pd.DataFrame,
    adaptation: pd.DataFrame,
    features: list[str],
) -> RandomForestRegressor:
    """
    Train a candidate using only base and adaptation engines.

    Strategies:
    - replay_1x: equal per-row weights for base and adaptation.
    - adaptation_2x: adaptation rows receive twice the weight.
    - adaptation_4x: adaptation rows receive four times the weight.
    - engine_balanced_2x: each engine has equal total base weight;
      adaptation engines receive twice the total weight of a base
      engine, irrespective of their number of rows.
    """

    combined = pd.concat(
        [base, adaptation],
        ignore_index=True,
    )

    base_count = len(base)
    adapt_count = len(adaptation)

    weights = np.ones(
        len(combined),
        dtype=float,
    )

    if strategy == "replay_1x":
        weights[base_count:] = 1.0

    elif strategy == "adaptation_2x":
        weights[base_count:] = 2.0

    elif strategy == "adaptation_4x":
        weights[base_count:] = 4.0

    elif strategy == "engine_balanced_2x":
        # Equal total weight for each base engine.
        base_engine_sizes = base.groupby(GROUP).size()
        base_engine_count = len(base_engine_sizes)

        for engine_id, row_count in base_engine_sizes.items():
            mask = (
                combined.index < base_count
            ) & (
                combined[GROUP] == engine_id
            )

            weights[mask] = (
                1.0 / row_count
            )

        # Each adaptation engine receives twice the total
        # weight of one base engine.
        adaptation_engine_sizes = (
            adaptation.groupby(GROUP).size()
        )

        for engine_id, row_count in adaptation_engine_sizes.items():
            mask = (
                combined.index >= base_count
            ) & (
                combined[GROUP] == engine_id
            )

            weights[mask] = (
                2.0 / row_count
            )

        # Rescale weights to have mean 1 for stable interpretation.
        weights *= (
            len(weights) / weights.sum()
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    if (
        len(weights) != len(combined)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Invalid sample weights were produced."
        )

    model = make_model()

    model.fit(
        combined[features],
        combined[TARGET],
        sample_weight=weights,
    )

    return model


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main() -> None:

    root = find_project_root()

    input_path = (
        root
        / "outputs"
        / "step01"
        / "train_FD001_processed.csv"
    )

    output_dir = (
        root
        / "outputs"
        / "step10E_evolution_diagnosis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("STEP 10E — SELF-EVOLUTION DIAGNOSIS")
    print("=" * 72)

    data = pd.read_csv(input_path)

    missing = REQUIRED_COLUMNS - set(data.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    if (
        data[GROUP].isna().any()
        or data[TARGET].isna().any()
        or data["cycle"].isna().any()
    ):
        raise ValueError(
            "Engine IDs, cycles, and RUL labels must not be missing."
        )

    features = [
        col for col in data.columns
        if (
            col not in NON_FEATURES
            and pd.api.types.is_numeric_dtype(data[col])
        )
    ]

    if data[features].isna().any().any():
        raise ValueError(
            "Numeric features contain missing values."
        )

    for sensor in SHIFT_SPEC:
        if sensor not in features:
            raise ValueError(
                f"Required sensor missing: {sensor}"
            )

    engines = np.array(
        sorted(data[GROUP].unique())
    )

    if len(engines) < 40:
        raise ValueError(
            "Too few engines for the planned split."
        )

    # --------------------------------------------------------
    # 9.1 Split at engine level
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(engines)

    n_base = int(
        math.floor(len(engines) * BASE_FRACTION)
    )

    n_adapt = int(
        math.floor(len(engines) * ADAPT_FRACTION)
    )

    n_gate = int(
        math.floor(len(engines) * GATE_FRACTION)
    )

    if n_base + n_adapt + n_gate >= len(engines):
        raise ValueError(
            "The split leaves no final evaluation engines."
        )

    base_ids = set(shuffled[:n_base])

    adapt_ids = set(
        shuffled[n_base:n_base + n_adapt]
    )

    gate_start = n_base + n_adapt

    gate_ids = set(
        shuffled[gate_start:gate_start + n_gate]
    )

    final_ids = set(
        shuffled[gate_start + n_gate:]
    )

    assert base_ids.isdisjoint(adapt_ids)
    assert base_ids.isdisjoint(gate_ids)
    assert base_ids.isdisjoint(final_ids)
    assert adapt_ids.isdisjoint(gate_ids)
    assert adapt_ids.isdisjoint(final_ids)
    assert gate_ids.isdisjoint(final_ids)

    base = data[data[GROUP].isin(base_ids)].copy()

    adaptation_raw = data[
        data[GROUP].isin(adapt_ids)
    ].copy()

    gate_raw = data[
        data[GROUP].isin(gate_ids)
    ].copy()

    final_raw = data[
        data[GROUP].isin(final_ids)
    ].copy()

    print(f"\nProject: {root}")
    print(f"Total rows: {len(data)}")
    print(f"Total engines: {len(engines)}")
    print(f"Numeric features: {len(features)}")
    print(f"Base engines: {len(base_ids)}")
    print(f"Adaptation engines: {len(adapt_ids)}")
    print(f"Gate-validation engines: {len(gate_ids)}")
    print(f"Final held-out engines: {len(final_ids)}")

    # --------------------------------------------------------
    # 9.2 Calculate shift scale using base engines only
    # --------------------------------------------------------

    base_stds = base[features].std(ddof=0)

    base_stds = base_stds.where(
        np.isfinite(base_stds) & (base_stds > 0),
        1.0,
    )

    adaptation = apply_shift(
        adaptation_raw,
        features,
        base_stds,
    )

    gate = apply_shift(
        gate_raw,
        features,
        base_stds,
    )

    final_eval = apply_shift(
        final_raw,
        features,
        base_stds,
    )

    print("\nSynthetic shift configuration:")
    for sensor, magnitude in SHIFT_SPEC.items():
        print(f"  {sensor}: {magnitude:+.2f} standard deviations")

    # --------------------------------------------------------
    # 9.3 Train frozen baseline
    # --------------------------------------------------------

    print("\nTraining frozen baseline...")

    frozen = make_model()

    frozen.fit(
        base[features],
        base[TARGET],
    )

    frozen_gate_metrics, frozen_gate_per_engine = (
        per_engine_metrics(
            frozen,
            gate,
            features,
            "frozen_gate",
        )
    )

    # --------------------------------------------------------
    # 9.4 Diagnose baseline validation errors
    # --------------------------------------------------------

    frozen_gate_per_engine.to_csv(
        output_dir / "step10E_gate_engine_errors_frozen.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9.5 Evaluate predefined candidate strategies
    # --------------------------------------------------------

    strategies = [
        "replay_1x",
        "adaptation_2x",
        "adaptation_4x",
        "engine_balanced_2x",
    ]

    candidate_models = {}
    candidate_gate_metrics = {}
    candidate_engine_errors = {}
    strategy_records = []

    print("\nEvaluating candidate update strategies...")

    for strategy in strategies:

        candidate = train_candidate(
            strategy,
            base,
            adaptation,
            features,
        )

        gate_metrics, gate_engine_errors = (
            per_engine_metrics(
                candidate,
                gate,
                features,
                f"{strategy}_gate",
            )
        )

        accepted, reason = gate_candidate(
            frozen_gate_metrics,
            gate_metrics,
        )

        candidate_models[strategy] = candidate
        candidate_gate_metrics[strategy] = gate_metrics
        candidate_engine_errors[strategy] = gate_engine_errors

        strategy_records.append({
            "strategy": strategy,
            **gate_metrics,
            "accepted_against_frozen_baseline": accepted,
            "gate_decision_reason": reason,
        })

        gate_engine_errors.to_csv(
            output_dir
            / f"step10E_gate_engine_errors_{strategy}.csv",
            index=False,
        )

        print(f"\nStrategy: {strategy}")
        print(
            f"  Mean engine RMSE: "
            f"{gate_metrics['mean_engine_RMSE']:.4f}"
        )
        print(
            f"  Mean engine MAE: "
            f"{gate_metrics['mean_engine_MAE']:.4f}"
        )
        print(
            f"  Mean engine NASA score: "
            f"{gate_metrics['mean_engine_NASA_score']:.4f}"
        )
        print(f"  {reason}")

    strategy_df = pd.DataFrame(strategy_records)

    strategy_path = (
        output_dir / "step10E_candidate_strategy_validation.csv"
    )

    strategy_df.to_csv(
        strategy_path,
        index=False,
    )

    # --------------------------------------------------------
    # 9.6 Select only among candidates passing the gate
    #
    # Selection uses gate-validation results only.
    # No final evaluation data is used here.
    # --------------------------------------------------------

    eligible = strategy_df[
        strategy_df["accepted_against_frozen_baseline"]
    ].copy()

    selected_strategy = None

    if not eligible.empty:
        # Fixed selection criterion:
        # equal-weight average of relative changes in the three
        # monitored mean engine-level metrics versus baseline.
        for metric in MONITORED_METRICS:
            baseline_value = frozen_gate_metrics[metric]

            eligible[f"relative_{metric}"] = (
                eligible[metric] - baseline_value
            ) / max(abs(baseline_value), 1e-12)

        eligible["selection_objective"] = (
            eligible[f"relative_{MONITORED_METRICS[0]}"]
            + eligible[f"relative_{MONITORED_METRICS[1]}"]
            + eligible[f"relative_{MONITORED_METRICS[2]}"]
        ) / 3.0

        selected_strategy = str(
            eligible.sort_values(
                "selection_objective"
            ).iloc[0]["strategy"]
        )

        selected_model = candidate_models[
            selected_strategy
        ]

        selection_status = (
            "candidate_accepted_and_selected"
        )

    else:
        selected_model = frozen

        selection_status = (
            "no_candidate_passed_gate_frozen_model_retained"
        )

    print("\n" + "=" * 72)
    print("VALIDATION SELECTION")
    print("=" * 72)

    if selected_strategy is None:
        print(
            "No candidate passed all predefined safety criteria."
        )
        print(
            "The frozen baseline remains the selected model."
        )
    else:
        print(
            f"Selected strategy: {selected_strategy}"
        )

    # --------------------------------------------------------
    # 9.7 Final evaluation exactly once after selection
    # --------------------------------------------------------

    print("\nFinal held-out evaluation...")

    final_models = {
        "frozen_baseline": frozen,
    }

    final_models.update({
        f"candidate_{name}": model
        for name, model in candidate_models.items()
    })

    final_models["selected_safety_gated_model"] = (
        selected_model
    )

    final_records = []
    final_engine_tables = []

    for model_name, model in final_models.items():

        aggregate, engine_metrics = per_engine_metrics(
            model,
            final_eval,
            features,
            model_name,
        )

        final_records.append({
            "model": model_name,
            **aggregate,
        })

        engine_metrics["model"] = model_name

        final_engine_tables.append(
            engine_metrics
        )

    final_metrics_df = pd.DataFrame(
        final_records
    )

    final_engine_df = pd.concat(
        final_engine_tables,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 9.8 Save selected-model feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": features,
        "frozen_feature_importance": (
            frozen.feature_importances_
        ),
        "selected_feature_importance": (
            selected_model.feature_importances_
        ),
    })

    importance_df["change"] = (
        importance_df["selected_feature_importance"]
        - importance_df["frozen_feature_importance"]
    )

    importance_df = importance_df.sort_values(
        "selected_feature_importance",
        ascending=False,
    )

    # --------------------------------------------------------
    # 9.9 Save reports
    # --------------------------------------------------------

    final_metrics_path = (
        output_dir / "step10E_final_metrics.csv"
    )

    final_engine_path = (
        output_dir / "step10E_final_engine_metrics.csv"
    )

    importance_path = (
        output_dir / "step10E_feature_importance_change.csv"
    )

    report_path = (
        output_dir / "step10E_report.json"
    )

    final_metrics_df.to_csv(
        final_metrics_path,
        index=False,
    )

    final_engine_df.to_csv(
        final_engine_path,
        index=False,
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    report = {
        "experiment": (
            "self_evolution_diagnosis_and_candidate_update_evaluation"
        ),
        "status": "completed",
        "dataset": str(input_path),
        "seed": SEED,
        "base_engines": len(base_ids),
        "adaptation_engines": len(adapt_ids),
        "gate_engines": len(gate_ids),
        "final_heldout_engines": len(final_ids),
        "candidate_strategies": strategies,
        "selected_strategy": selected_strategy,
        "selection_status": selection_status,
        "synthetic_sensor_shift": SHIFT_SPEC,
        "gate_criteria": {
            "max_allowed_degradation": (
                MAX_ALLOWED_DEGRADATION
            ),
            "minimum_required_improvement": (
                MIN_REQUIRED_IMPROVEMENT
            ),
            "metrics": MONITORED_METRICS,
            "aggregation": (
                "Mean of per-engine metrics"
            ),
        },
        "final_evaluation_used_for_selection": False,
        "official_NASA_test_file_used": False,
        "limitations": [
            "The sensor shift is synthetic.",
            "The gate-validation set contains a limited number of engines.",
            "Candidate strategies are selected using the same gate set, so validation results may be optimistic.",
            "The final held-out evaluation is separate and must remain untouched.",
            "Random Forest feature importance is not causal.",
            "This experiment evaluates controlled model updates, not unrestricted autonomous learning.",
            "One split does not establish generalization; repeat with prespecified engine splits and shift scenarios.",
        ],
        "outputs": [
            str(strategy_path),
            str(final_metrics_path),
            str(final_engine_path),
            str(importance_path),
            str(report_path),
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 9.10 Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 10E — FINAL HELD-OUT RESULTS")
    print("=" * 72)

    print(
        final_metrics_df[
            [
                "model",
                "n_engines",
                "mean_engine_RMSE",
                "mean_engine_MAE",
                "mean_engine_NASA_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nOutput directory:")
    print(output_dir)

    print("\nExperiment finished.")
    print(
        "Do not change the acceptance criteria after seeing "
        "final held-out results."
    )


# ============================================================
# 10. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
