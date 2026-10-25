
"""
STEP 10D — SAFETY-GATED SELF-EVOLUTION EXPERIMENT
NASA C-MAPSS FD001 | Python 3.13+

Compares:
1. Frozen baseline model.
2. Unconditionally retrained model.
3. Safety-gated self-evolved model.

The sensor shift is synthetic, not observed real-world drift.
Only the processed NASA training file is used.
The official NASA test file is not used.

Leakage controls:
- Split at engine level before selecting rows.
- Keep base, adaptation, gate-validation and final engines disjoint.
- Use gate-validation engines for update acceptance only.
- Keep final evaluation engines untouched until the end.
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
# 1. FIXED CONFIGURATION
# ============================================================

RANDOM_SEED = 20261025
SHIFT_SEED = 4107

BASE_ENGINE_FRACTION = 0.60
ADAPT_ENGINE_FRACTION = 0.15
GATE_ENGINE_FRACTION = 0.10

N_ADAPTATION_ROUNDS = 3
N_TREES = 100
MIN_SAMPLES_LEAF = 2
MAX_DEPTH = 18

# Candidate update must not worsen any monitored metric by
# more than 2% on either validation slice.
MAX_ALLOWED_RELATIVE_DEGRADATION = 0.02

# At least one snapshot metric must improve by at least 1%.
MIN_REQUIRED_SNAPSHOT_IMPROVEMENT = 0.01

TARGET = "RUL"
GROUP = "unit_id"

REQUIRED_COLUMNS = {GROUP, "cycle", TARGET}
NON_FEATURES = {TARGET, GROUP}

# Predefined synthetic shift, measured in standard deviations
# computed using base-training data only.
SHIFT_SPEC = {
    "sensor_04": +0.60,
    "sensor_07": -0.60,
    "sensor_11": +0.80,
    "sensor_12": -0.70,
    "sensor_15": +0.65,
}

MONITORED_METRICS = (
    "RMSE",
    "MAE",
    "NASA_PHM_score",
)


# ============================================================
# 2. FIND PROJECT ROOT
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
        "Cannot find outputs\\step01\\train_FD001_processed.csv. "
        "Save this script in the NASA project's src folder."
    )


# ============================================================
# 3. MODEL FACTORY
# ============================================================

def create_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ============================================================
# 4. APPLY SYNTHETIC SENSOR SHIFT
# ============================================================

def apply_synthetic_shift(
    frame: pd.DataFrame,
    feature_columns: list[str],
    base_stds: pd.Series,
) -> pd.DataFrame:
    shifted = frame.copy()

    for sensor, shift_units in SHIFT_SPEC.items():
        if sensor not in feature_columns:
            raise ValueError(
                f"Shift sensor missing from feature set: {sensor}"
            )

        scale = float(base_stds[sensor])

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Invalid base-training standard deviation "
                f"for {sensor}: {scale}"
            )

        shifted[sensor] = (
            shifted[sensor] + shift_units * scale
        )

    return shifted


# ============================================================
# 5. NASA PHM SCORE
# ============================================================

def nasa_score(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Asymmetric NASA PHM score.
    Lower is better.
    """

    error = (
        np.asarray(predicted, dtype=float)
        - np.asarray(actual, dtype=float)
    )

    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return float(np.sum(penalties))


# ============================================================
# 6. METRICS
# ============================================================

def metrics_for_rows(
    model: RandomForestRegressor,
    frame: pd.DataFrame,
    features: list[str],
    label: str,
) -> dict:
    actual = frame[TARGET].to_numpy(dtype=float)

    predicted = model.predict(
        frame[features]
    )

    return {
        "evaluation_slice": label,
        "n_rows": int(len(frame)),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "NASA_PHM_score": nasa_score(
            actual,
            predicted,
        ),
        "mean_error_pred_minus_actual": float(
            np.mean(predicted - actual)
        ),
    }


# ============================================================
# 7. SELECT SNAPSHOTS PER ENGINE
# ============================================================

def snapshot_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one observed row near 50%, 70%, and 90% of each
    engine's observed cycle range.

    These are evaluation snapshots. Their RUL labels are used
    only for evaluation, never for feature construction.
    """

    selected = []

    for _, engine_data in frame.groupby(
        GROUP,
        sort=False,
    ):
        engine_data = engine_data.sort_values(
            "cycle"
        )

        cycles = engine_data[
            "cycle"
        ].to_numpy()

        min_cycle = float(np.min(cycles))
        max_cycle = float(np.max(cycles))
        span = max_cycle - min_cycle

        for fraction in (0.50, 0.70, 0.90):
            target_cycle = (
                min_cycle + fraction * span
            )

            index = int(
                np.argmin(
                    np.abs(cycles - target_cycle)
                )
            )

            selected.append(
                engine_data.iloc[index]
            )

    if not selected:
        raise ValueError(
            "No rows available for snapshot evaluation."
        )

    return pd.DataFrame(
        selected
    ).reset_index(drop=True)


# ============================================================
# 8. EVALUATE MODEL ON TWO SLICES
# ============================================================

def evaluate_bundle(
    model: RandomForestRegressor,
    frame: pd.DataFrame,
    features: list[str],
    prefix: str,
) -> dict[str, dict]:
    snapshots = snapshot_rows(frame)

    return {
        "all_cycles": metrics_for_rows(
            model,
            frame,
            features,
            f"{prefix}_all_cycles",
        ),
        "snapshots": metrics_for_rows(
            model,
            snapshots,
            features,
            f"{prefix}_50_70_90pct_life_snapshots",
        ),
    }


# ============================================================
# 9. SAFETY-GATE DECISION
# ============================================================

def decide_update(
    incumbent_metrics: dict[str, dict],
    candidate_metrics: dict[str, dict],
) -> tuple[bool, str]:
    """
    Accept only if:
    1. No monitored metric worsens by more than the predefined
       tolerance on either all-cycle or snapshot validation.
    2. At least one snapshot metric improves by the predefined
       minimum amount.

    Otherwise reject and retain the incumbent model.
    """

    violations = []
    improvements = []

    for slice_name in (
        "all_cycles",
        "snapshots",
    ):
        incumbent = incumbent_metrics[
            slice_name
        ]

        candidate = candidate_metrics[
            slice_name
        ]

        for metric in MONITORED_METRICS:
            old_value = float(
                incumbent[metric]
            )

            new_value = float(
                candidate[metric]
            )

            allowed_value = (
                old_value
                * (1.0 + MAX_ALLOWED_RELATIVE_DEGRADATION)
            )

            if new_value > allowed_value:
                violations.append(
                    f"{slice_name}.{metric} exceeded "
                    f"the allowed degradation "
                    f"({old_value:.6f} -> {new_value:.6f})"
                )

            if (
                slice_name == "snapshots"
                and old_value > 0
            ):
                relative_improvement = (
                    old_value - new_value
                ) / old_value

                if (
                    relative_improvement
                    >= MIN_REQUIRED_SNAPSHOT_IMPROVEMENT
                ):
                    improvements.append(
                        f"snapshot {metric} improved "
                        f"by {relative_improvement:.2%}"
                    )

    if violations:
        return (
            False,
            "Rejected: " + "; ".join(violations),
        )

    if not improvements:
        return (
            False,
            "Rejected: no snapshot metric improved by at least "
            f"{MIN_REQUIRED_SNAPSHOT_IMPROVEMENT:.1%}.",
        )

    return (
        True,
        "Accepted: " + "; ".join(improvements),
    )


# ============================================================
# 10. MAIN EXPERIMENT
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
        / "step10D_safety_gated_evolution"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("STEP 10D — SAFETY-GATED SELF-EVOLUTION")
    print("=" * 72)
    print(f"Project root: {root}")
    print(f"Input: {input_path}")

    # --------------------------------------------------------
    # 10.1 Load and validate data
    # --------------------------------------------------------

    data = pd.read_csv(input_path)

    missing = (
        REQUIRED_COLUMNS - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if (
        data[GROUP].isna().any()
        or data[TARGET].isna().any()
    ):
        raise ValueError(
            "Engine identifiers and RUL labels "
            "must not be missing."
        )

    if data["cycle"].isna().any():
        raise ValueError(
            "Cycle values must not be missing."
        )

    features = [
        column
        for column in data.columns
        if (
            column not in NON_FEATURES
            and pd.api.types.is_numeric_dtype(
                data[column]
            )
        )
    ]

    if not features:
        raise ValueError(
            "No numeric model features found."
        )

    if data[features].isna().any().any():
        raise ValueError(
            "Numeric features contain missing values."
        )

    for sensor in SHIFT_SPEC:
        if sensor not in features:
            raise ValueError(
                f"Configured shift sensor missing: {sensor}"
            )

    engines = np.array(
        sorted(data[GROUP].unique())
    )

    if len(engines) < 30:
        raise ValueError(
            "At least 30 engines are required for this split. "
            f"Found {len(engines)}."
        )

    print(f"Total rows: {len(data)}")
    print(f"Total engines: {len(engines)}")
    print(f"Numeric features: {len(features)}")

    # --------------------------------------------------------
    # 10.2 Split engines into four disjoint groups
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    shuffled = rng.permutation(
        engines
    )

    n_base = int(
        math.floor(
            len(engines) * BASE_ENGINE_FRACTION
        )
    )

    n_adapt = int(
        math.floor(
            len(engines) * ADAPT_ENGINE_FRACTION
        )
    )

    n_gate = int(
        math.floor(
            len(engines) * GATE_ENGINE_FRACTION
        )
    )

    n_adapt = max(
        N_ADAPTATION_ROUNDS,
        n_adapt,
    )

    if n_base + n_adapt + n_gate >= len(engines):
        raise ValueError(
            "Engine split leaves no final evaluation engines."
        )

    base_ids = set(
        shuffled[:n_base]
    )

    adapt_ids = list(
        shuffled[n_base:n_base + n_adapt]
    )

    gate_start = n_base + n_adapt

    gate_ids = set(
        shuffled[gate_start:gate_start + n_gate]
    )

    final_ids = set(
        shuffled[gate_start + n_gate:]
    )

    base = data[
        data[GROUP].isin(base_ids)
    ].copy()

    adaptation_raw = data[
        data[GROUP].isin(adapt_ids)
    ].copy()

    gate_raw = data[
        data[GROUP].isin(gate_ids)
    ].copy()

    final_raw = data[
        data[GROUP].isin(final_ids)
    ].copy()

    # Verify engine-level separation.
    assert base_ids.isdisjoint(set(adapt_ids))
    assert base_ids.isdisjoint(gate_ids)
    assert base_ids.isdisjoint(final_ids)
    assert set(adapt_ids).isdisjoint(gate_ids)
    assert set(adapt_ids).isdisjoint(final_ids)
    assert gate_ids.isdisjoint(final_ids)

    print("\nEngine-level split")
    print("-" * 40)
    print(f"Base-training engines: {len(base_ids)}")
    print(f"Adaptation engines: {len(adapt_ids)}")
    print(f"Gate-validation engines: {len(gate_ids)}")
    print(f"Final held-out engines: {len(final_ids)}")

    # --------------------------------------------------------
    # 10.3 Calculate shift scales using base data only
    # --------------------------------------------------------

    base_stds = base[features].std(
        ddof=0
    )

    base_stds = base_stds.where(
        np.isfinite(base_stds)
        & (base_stds > 0),
        1.0,
    )

    adaptation = apply_synthetic_shift(
        adaptation_raw,
        features,
        base_stds,
    )

    gate = apply_synthetic_shift(
        gate_raw,
        features,
        base_stds,
    )

    final_eval = apply_synthetic_shift(
        final_raw,
        features,
        base_stds,
    )

    print("\nSynthetic sensor shift")
    print("-" * 40)

    for sensor, magnitude in SHIFT_SPEC.items():
        print(
            f"{sensor}: {magnitude:+.2f} standard deviations"
        )

    # --------------------------------------------------------
    # 10.4 Train frozen baseline
    # --------------------------------------------------------

    print("\nTraining frozen baseline...")

    frozen = create_model()

    frozen.fit(
        base[features],
        base[TARGET],
    )

    # Initial incumbent starts as the frozen baseline.
    incumbent = create_model()

    incumbent.fit(
        base[features],
        base[TARGET],
    )

    incumbent_gate_metrics = evaluate_bundle(
        incumbent,
        gate,
        features,
        "incumbent_before_updates",
    )

    # --------------------------------------------------------
    # 10.5 Sequential adaptation with safety gate
    # --------------------------------------------------------

    replay = base.copy()

    update_history = []

    adaptation_order = (
        np.random.default_rng(SHIFT_SEED)
        .permutation(np.array(adapt_ids))
    )

    rounds = np.array_split(
        adaptation_order,
        N_ADAPTATION_ROUNDS,
    )

    accepted_updates = 0

    print("\nSequential adaptation with safety gate")
    print("-" * 40)

    for round_number, engine_group in enumerate(
        rounds,
        start=1,
    ):
        batch_ids = set(
            engine_group.tolist()
        )

        batch = adaptation[
            adaptation[GROUP].isin(batch_ids)
        ].copy()

        if batch.empty:
            raise RuntimeError(
                f"Adaptation round {round_number} is empty."
            )

        # Labelled outcomes from this batch become available.
        # Even if a model update is rejected, the labelled batch
        # remains available for subsequent candidate training.
        replay = pd.concat(
            [replay, batch],
            ignore_index=True,
        )

        candidate = create_model()

        candidate.fit(
            replay[features],
            replay[TARGET],
        )

        candidate_gate_metrics = evaluate_bundle(
            candidate,
            gate,
            features,
            f"candidate_round_{round_number}",
        )

        accepted, reason = decide_update(
            incumbent_gate_metrics,
            candidate_gate_metrics,
        )

        if accepted:
            incumbent = candidate
            incumbent_gate_metrics = (
                candidate_gate_metrics
            )

            accepted_updates += 1
            decision = "ACCEPTED"

        else:
            # Rollback: keep the previous accepted model.
            decision = (
                "REJECTED — previous accepted model retained"
            )

        update_history.append({
            "round": round_number,
            "new_adaptation_engines": len(batch_ids),
            "newly_labelled_rows": int(len(batch)),
            "replay_rows": int(len(replay)),
            "decision": decision,
            "accepted": bool(accepted),
            "reason": reason,
            "gate_incumbent_all_cycles_RMSE": float(
                incumbent_gate_metrics[
                    "all_cycles"
                ]["RMSE"]
            ),
            "gate_incumbent_all_cycles_MAE": float(
                incumbent_gate_metrics[
                    "all_cycles"
                ]["MAE"]
            ),
            "gate_incumbent_all_cycles_NASA_PHM_score": float(
                incumbent_gate_metrics[
                    "all_cycles"
                ]["NASA_PHM_score"]
            ),
            "gate_incumbent_snapshots_RMSE": float(
                incumbent_gate_metrics[
                    "snapshots"
                ]["RMSE"]
            ),
            "gate_incumbent_snapshots_MAE": float(
                incumbent_gate_metrics[
                    "snapshots"
                ]["MAE"]
            ),
            "gate_incumbent_snapshots_NASA_PHM_score": float(
                incumbent_gate_metrics[
                    "snapshots"
                ]["NASA_PHM_score"]
            ),
            "final_evaluation_used_for_decision": False,
        })

        print(
            f"\nRound {round_number}/{N_ADAPTATION_ROUNDS}: "
            f"{decision}"
        )

        print(f"  {reason}")

    # --------------------------------------------------------
    # 10.6 Create the three comparison models
    # --------------------------------------------------------

    unconditional = create_model()

    unconditional.fit(
        replay[features],
        replay[TARGET],
    )

    final_models = {
        "frozen_baseline": frozen,
        "unconditionally_retrained_after_all_batches": (
            unconditional
        ),
        "safety_gated_self_evolved": incumbent,
    }

    # --------------------------------------------------------
    # 10.7 Final held-out evaluation
    #
    # Final engines have not been used to train, update,
    # select, or accept any model.
    # --------------------------------------------------------

    print("\nEvaluating on final held-out engines...")

    final_metrics_records = []

    for model_name, model in final_models.items():

        bundle = evaluate_bundle(
            model,
            final_eval,
            features,
            model_name,
        )

        for slice_name, result in bundle.items():
            record = dict(result)

            record["model"] = model_name
            record["evaluation_slice"] = slice_name
            record["n_heldout_engines"] = len(final_ids)

            final_metrics_records.append(
                record
            )

    final_metrics = pd.DataFrame(
        final_metrics_records
    )

    # --------------------------------------------------------
    # 10.8 Compare global feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": features,
        "frozen_feature_importance": (
            frozen.feature_importances_
        ),
        "safety_gated_feature_importance": (
            incumbent.feature_importances_
        ),
    })

    importance_df[
        "change_gated_minus_frozen"
    ] = (
        importance_df[
            "safety_gated_feature_importance"
        ]
        - importance_df[
            "frozen_feature_importance"
        ]
    )

    importance_df = importance_df.sort_values(
        "safety_gated_feature_importance",
        ascending=False,
    )

    # --------------------------------------------------------
    # 10.9 Save outputs
    # --------------------------------------------------------

    metrics_path = (
        output_dir / "step10D_final_metrics.csv"
    )

    history_path = (
        output_dir / "step10D_update_decisions.csv"
    )

    importance_path = (
        output_dir
        / "step10D_feature_importance_change.csv"
    )

    report_path = (
        output_dir / "step10D_report.json"
    )

    final_metrics.to_csv(
        metrics_path,
        index=False,
    )

    pd.DataFrame(
        update_history
    ).to_csv(
        history_path,
        index=False,
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    report = {
        "experiment": (
            "safety_gated_self_evolution_under_synthetic_sensor_shift"
        ),
        "status": "completed",
        "dataset": str(input_path),
        "random_seed": RANDOM_SEED,
        "synthetic_shift_seed": SHIFT_SEED,
        "base_engine_count": len(base_ids),
        "adaptation_engine_count": len(adapt_ids),
        "gate_validation_engine_count": len(gate_ids),
        "final_heldout_engine_count": len(final_ids),
        "adaptation_rounds": N_ADAPTATION_ROUNDS,
        "accepted_updates": accepted_updates,
        "rejected_updates": (
            N_ADAPTATION_ROUNDS - accepted_updates
        ),
        "safety_gate": {
            "max_allowed_relative_degradation": (
                MAX_ALLOWED_RELATIVE_DEGRADATION
            ),
            "minimum_required_snapshot_improvement": (
                MIN_REQUIRED_SNAPSHOT_IMPROVEMENT
            ),
            "monitored_metrics": list(
                MONITORED_METRICS
            ),
            "evaluation_slices": [
                "all_cycles",
                "50/70/90-percent life snapshots",
            ],
        },
        "synthetic_sensor_shift_standard_deviation_units": (
            SHIFT_SPEC
        ),
        "engine_groups_disjoint": True,
        "final_evaluation_used_for_update_or_selection": False,
        "official_NASA_test_file_used": False,
        "interpretation_limits": [
            "The sensor shift is synthetic, not observed real-world drift.",
            "Gate validation uses a limited number of engines.",
            "Accepted updates do not prove deployment safety.",
            "Rows within an engine are dependent.",
            "All-cycle rows must not be treated as independent engines.",
            "Random Forest feature importance is not causal or local SHAP.",
            "The experiment demonstrates constrained updating and rollback, not unrestricted autonomous learning.",
            "Rejecting every update is a valid outcome if the candidates fail the predefined gate.",
        ],
        "outputs": [
            str(metrics_path),
            str(history_path),
            str(importance_path),
            str(report_path),
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 10.10 Print results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 10D FINAL HELD-OUT RESULTS")
    print("=" * 72)

    columns_to_print = [
        "model",
        "evaluation_slice",
        "n_heldout_engines",
        "n_rows",
        "RMSE",
        "MAE",
        "NASA_PHM_score",
    ]

    print(
        final_metrics[columns_to_print].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nAccepted updates: "
        f"{accepted_updates}/{N_ADAPTATION_ROUNDS}"
    )

    print("\nSaved output files:")

    for path in [
        metrics_path,
        history_path,
        importance_path,
        report_path,
    ]:
        print(path)

    print("\nExperiment finished.")
    print(
        "Compare all three models on the held-out engines. "
        "Do not claim that the safety gate is superior unless "
        "the final metrics support that conclusion."
    )


# ============================================================
# 11. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise
