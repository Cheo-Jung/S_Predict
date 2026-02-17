"""Predict latest direction using saved artifacts and real OHLCV refresh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_mvp import (
    apply_strategy,
    build_dataset,
    fetch_yahoo_ohlcv,
    generate_synthetic_data,
    load_csv,
    standardize_apply,
)

MODEL_PATH = Path("artifacts/model.json")


def resolve_rows(model: dict, csv_path: Path | None, symbol: str | None, start: str, end: str):
    if csv_path:
        return load_csv(csv_path), f"csv:{csv_path}"

    src = model.get("source", "yahoo")
    sym = symbol or model.get("symbol", "AAPL")
    if src == "yahoo":
        try:
            return fetch_yahoo_ohlcv(sym, start, end, "1d"), f"yahoo:{sym}"
        except Exception:
            return generate_synthetic_data(seed=7), "synthetic:fallback"

    return generate_synthetic_data(seed=7), "synthetic:fallback"


def predict_latest(csv_path: Path | None = None, symbol: str | None = None, start: str = "2016-01-01", end: str = "2100-01-01") -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing artifacts/model.json. Run training first.")

    model = json.loads(MODEL_PATH.read_text())
    rows, data_source = resolve_rows(model, csv_path, symbol, start, end)
    x_raw, _, _ = build_dataset(rows)
    x_std = standardize_apply([x_raw[-1]], model["means"], model["stds"])

    probs = apply_strategy(model["selected_strategy"], [x_raw[-1]], x_std, model["ensemble_models"])
    p_up = probs[0]
    thr = float(model.get("tuned_threshold", 0.5))

    if p_up >= thr:
        signal = "UP"
    elif p_up <= (1.0 - thr):
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    return {
        "task": "direction",
        "symbol": symbol or model.get("symbol", "AAPL"),
        "data_source": data_source,
        "selected_strategy": model["selected_strategy"],
        "prob_up": p_up,
        "decision_threshold": thr,
        "predicted_direction": signal,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2100-01-01")
    args = p.parse_args()
    r = predict_latest(args.csv, args.symbol, args.start, args.end)
    print(
        f"{r['symbol']} => {r['predicted_direction']} "
        f"(P(up)={r['prob_up']:.4f}, thr={r['decision_threshold']:.2f}, strategy={r['selected_strategy']}, source={r['data_source']})"
    )


if __name__ == "__main__":
    main()
