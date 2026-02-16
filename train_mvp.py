"""Dependency-light MVP trainer for stock return/direction prediction.

Improvements focused on directional accuracy:
- classification objective (logistic SGD) for direction task
- richer handcrafted features
- thresholded directional evaluation (coverage + hit rate)
- walk-forward directional validation

This script can train from:
1) local CSV file with columns: close,volume
2) generated synthetic OHLCV-like data when no CSV is provided
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.json"
META_PATH = ARTIFACT_DIR / "meta.json"


@dataclass
class Row:
    close: float
    volume: float


def generate_synthetic_data(days: int = 1500, seed: int = 42) -> list[Row]:
    random.seed(seed)
    price = 100.0
    prev_ret = 0.0
    data: list[Row] = []
    for _ in range(days):
        drift = 0.0004
        noise = random.gauss(0, 0.010)
        # mild auto-correlation to emulate weak momentum regimes
        ret = drift + 0.18 * prev_ret + noise
        price = max(1.0, price * (1 + ret))
        volume = 1_000_000 + random.randint(-150_000, 150_000)
        data.append(Row(close=price, volume=float(volume)))
        prev_ret = ret
    return data


def load_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"close", "volume"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include columns: close, volume")
        for rec in reader:
            rows.append(Row(close=float(rec["close"]), volume=float(rec["volume"])))
    if len(rows) < 250:
        raise ValueError("Need at least 250 rows for training/evaluation")
    return rows


def pct_change(values: list[float], lag: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for i in range(lag, len(values)):
        prev = values[i - lag]
        out[i] = (values[i] / prev - 1.0) if prev else None
    return out


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        if i >= window - 1:
            out[i] = running / window
    return out


def rolling_std(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        mean = sum(chunk) / window
        var = sum((x - mean) ** 2 for x in chunk) / window
        out[i] = math.sqrt(var)
    return out


def build_dataset(rows: list[Row]) -> tuple[list[list[float]], list[float], list[int]]:
    close = [r.close for r in rows]
    volume = [r.volume for r in rows]

    ret1 = pct_change(close, 1)
    ret3 = pct_change(close, 3)
    ret5 = pct_change(close, 5)
    ret10 = pct_change(close, 10)
    ma5 = rolling_mean(close, 5)
    ma20 = rolling_mean(close, 20)
    vol10 = rolling_std([r if r is not None else 0.0 for r in pct_change(close, 1)], 10)
    vol20 = rolling_std([r if r is not None else 0.0 for r in pct_change(close, 1)], 20)
    vol_chg_5 = pct_change(volume, 5)

    features: list[list[float]] = []
    target_ret: list[float] = []
    target_dir: list[int] = []

    for i in range(len(rows) - 1):
        vals = [ret1[i], ret3[i], ret5[i], ret10[i], ma5[i], ma20[i], vol10[i], vol20[i], vol_chg_5[i]]
        if any(v is None for v in vals):
            continue

        next_ret = close[i + 1] / close[i] - 1.0
        features.append(
            [
                float(ret1[i]),
                float(ret3[i]),
                float(ret5[i]),
                float(ret10[i]),
                float(ma5[i] / close[i] - 1.0),
                float(ma20[i] / close[i] - 1.0),
                float((ma5[i] - ma20[i]) / close[i]),
                float(vol10[i]),
                float(vol20[i]),
                float(volume[i] / 1_000_000.0),
                float(vol_chg_5[i]),
            ]
        )
        target_ret.append(next_ret)
        target_dir.append(1 if next_ret >= 0 else 0)

    return features, target_ret, target_dir


def standardize_fit(features: list[list[float]]) -> tuple[list[float], list[float]]:
    n_features = len(features[0])
    means = [sum(row[j] for row in features) / len(features) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        var = sum((row[j] - means[j]) ** 2 for row in features) / len(features)
        std = math.sqrt(var)
        stds.append(std if std > 1e-12 else 1.0)
    return means, stds


def standardize_apply(features: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in features]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_logistic_sgd(features: list[list[float]], labels: list[int], lr: float = 0.03, epochs: int = 300) -> tuple[list[float], float]:
    n_features = len(features[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        for xi, yi in zip(features, labels):
            z = bias + sum(w * x for w, x in zip(weights, xi))
            p = sigmoid(z)
            err = p - yi
            bias -= lr * err
            for j in range(n_features):
                weights[j] -= lr * err * xi[j]
    return weights, bias


def fit_linear_sgd(features: list[list[float]], target: list[float], lr: float = 0.03, epochs: int = 300) -> tuple[list[float], float]:
    n_features = len(features[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        for xi, yi in zip(features, target):
            pred = bias + sum(w * x for w, x in zip(weights, xi))
            err = pred - yi
            bias -= lr * err
            for j in range(n_features):
                weights[j] -= lr * err * xi[j]
    return weights, bias


def predict_many_linear(features: Iterable[list[float]], weights: list[float], bias: float) -> list[float]:
    return [bias + sum(w * x for w, x in zip(weights, xi)) for xi in features]


def predict_many_proba(features: Iterable[list[float]], weights: list[float], bias: float) -> list[float]:
    return [sigmoid(bias + sum(w * x for w, x in zip(weights, xi))) for xi in features]


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def directional_accuracy_from_sign(y_true: list[float], y_pred: list[float]) -> float:
    correct = 0
    for a, b in zip(y_true, y_pred):
        if (a >= 0 and b >= 0) or (a < 0 and b < 0):
            correct += 1
    return correct / len(y_true)


def directional_accuracy_cls(y_true: list[int], proba: list[float], threshold: float = 0.5) -> float:
    preds = [1 if p >= threshold else 0 for p in proba]
    correct = sum(int(a == b) for a, b in zip(y_true, preds))
    return correct / len(y_true)


def thresholded_hit_rate(y_true: list[int], proba: list[float], upper: float = 0.55, lower: float = 0.45) -> tuple[float, float]:
    selected_idx = [i for i, p in enumerate(proba) if p >= upper or p <= lower]
    if not selected_idx:
        return 0.0, 0.0
    correct = 0
    for i in selected_idx:
        pred = 1 if proba[i] >= upper else 0
        if pred == y_true[i]:
            correct += 1
    coverage = len(selected_idx) / len(y_true)
    hit_rate = correct / len(selected_idx)
    return coverage, hit_rate


def walk_forward_directional_accuracy(
    x: list[list[float]],
    y_dir: list[int],
    train_window: int = 600,
    test_window: int = 100,
) -> float:
    scores: list[float] = []
    start = 0
    while start + train_window + test_window <= len(x):
        x_tr = x[start : start + train_window]
        y_tr = y_dir[start : start + train_window]
        x_te = x[start + train_window : start + train_window + test_window]
        y_te = y_dir[start + train_window : start + train_window + test_window]

        means, stds = standardize_fit(x_tr)
        x_tr_s = standardize_apply(x_tr, means, stds)
        x_te_s = standardize_apply(x_te, means, stds)

        w, b = fit_logistic_sgd(x_tr_s, y_tr, lr=0.03, epochs=200)
        p = predict_many_proba(x_te_s, w, b)
        scores.append(directional_accuracy_cls(y_te, p, threshold=0.5))
        start += test_window

    return sum(scores) / len(scores) if scores else 0.0


def train(csv_path: Path | None = None, task: str = "direction", threshold: float = 0.55) -> dict:
    rows = load_csv(csv_path) if csv_path else generate_synthetic_data()
    x, y_ret, y_dir = build_dataset(rows)
    cut = int(len(x) * 0.8)

    x_train, x_test = x[:cut], x[cut:]
    y_train_ret, y_test_ret = y_ret[:cut], y_ret[cut:]
    y_train_dir, y_test_dir = y_dir[:cut], y_dir[cut:]

    means, stds = standardize_fit(x_train)
    x_train_s = standardize_apply(x_train, means, stds)
    x_test_s = standardize_apply(x_test, means, stds)

    model: dict
    metrics: dict[str, float | int | str]

    if task == "direction":
        weights, bias = fit_logistic_sgd(x_train_s, y_train_dir)
        proba = predict_many_proba(x_test_s, weights, bias)
        signed_pred = [1.0 if p >= 0.5 else -1.0 for p in proba]
        pseudo_ret = [s * 0.001 for s in signed_pred]
        coverage, hit_rate = thresholded_hit_rate(y_test_dir, proba, upper=threshold, lower=1 - threshold)

        metrics = {
            "task": "direction",
            "directional_accuracy": directional_accuracy_cls(y_test_dir, proba, threshold=0.5),
            "walk_forward_directional_accuracy": walk_forward_directional_accuracy(x, y_dir),
            "threshold": threshold,
            "threshold_coverage": coverage,
            "threshold_hit_rate": hit_rate,
            "mae_proxy": mae(y_test_ret, pseudo_ret),
            "rmse_proxy": rmse(y_test_ret, pseudo_ret),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "data_source": str(csv_path) if csv_path else "synthetic",
        }
        model = {"task": "direction", "weights": weights, "bias": bias, "means": means, "stds": stds}
    else:
        weights, bias = fit_linear_sgd(x_train_s, y_train_ret)
        pred = predict_many_linear(x_test_s, weights, bias)
        metrics = {
            "task": "return",
            "mae": mae(y_test_ret, pred),
            "rmse": rmse(y_test_ret, pred),
            "directional_accuracy": directional_accuracy_from_sign(y_test_ret, pred),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "data_source": str(csv_path) if csv_path else "synthetic",
        }
        model = {"task": "return", "weights": weights, "bias": bias, "means": means, "stds": stds}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    META_PATH.write_text(json.dumps(metrics, indent=2))

    print("[MVP] Evaluation metrics")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.6f}")
        else:
            print(f"- {k}: {v}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved meta  -> {META_PATH}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None, help="Optional OHLCV CSV path")
    parser.add_argument("--task", choices=["direction", "return"], default="direction")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold for hit-rate report")
    args = parser.parse_args()
    train(args.csv, task=args.task, threshold=args.threshold)


if __name__ == "__main__":
    main()
