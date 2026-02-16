"""MVP batch predictor based on saved model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_mvp import (
    build_dataset,
    generate_synthetic_data,
    load_csv,
    predict_many_linear,
    predict_many_proba,
    standardize_apply,
)

MODEL_PATH = Path("artifacts/model.json")


def predict_latest(csv_path: Path | None = None) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing artifacts/model.json. Run `python train_mvp.py` first.")

    model = json.loads(MODEL_PATH.read_text())
    rows = load_csv(csv_path) if csv_path else generate_synthetic_data(seed=7)
    x, _, _ = build_dataset(rows)
    x_latest = [x[-1]]
    x_latest = standardize_apply(x_latest, model["means"], model["stds"])

    if model.get("task") == "direction":
        proba_up = predict_many_proba(x_latest, model["weights"], model["bias"])[0]
        return {
            "task": "direction",
            "prob_up": proba_up,
            "predicted_direction": "UP" if proba_up >= 0.5 else "DOWN",
        }

    yhat = predict_many_linear(x_latest, model["weights"], model["bias"])[0]
    return {"task": "return", "predicted_next_day_return": yhat}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None, help="Optional OHLCV CSV path")
    args = parser.parse_args()
    result = predict_latest(args.csv)
    if result["task"] == "direction":
        print(
            f"Predicted direction: {result['predicted_direction']} "
            f"(P(up)={result['prob_up']:.4f})"
        )
    else:
        print(f"Predicted next-day return: {result['predicted_next_day_return']:.6%}")


if __name__ == "__main__":
    main()
