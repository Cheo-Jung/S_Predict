"""Real-data-first stock/crypto direction predictor (dependency-light).

Key goals:
- Use real OHLCV when available (Yahoo Finance HTTP API, or CSV input)
- Engineer practical technical features from OHLCV
- Train/evaluate direction model with time-aware split
- Persist artifacts for CLI/API inference
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.json"
META_PATH = ARTIFACT_DIR / "meta.json"


@dataclass
class Row:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_date_to_epoch(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def generate_synthetic_data(days: int = 1800, seed: int = 42) -> list[Row]:
    random.seed(seed)
    px = 100.0
    prev_ret = 0.0
    t0 = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp())
    out: list[Row] = []
    for i in range(days):
        noise = random.gauss(0, 0.01)
        ret = 0.0003 + 0.18 * prev_ret + noise
        op = px
        cl = max(1.0, op * (1 + ret))
        amp = abs(random.gauss(0.006, 0.003))
        hi = max(op, cl) * (1 + amp)
        lo = min(op, cl) * (1 - amp)
        vol = float(1_000_000 + random.randint(-220_000, 220_000))
        out.append(Row(ts=t0 + i * 86400, open=op, high=hi, low=lo, close=cl, volume=vol))
        px, prev_ret = cl, ret
    return out


def load_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include: timestamp,open,high,low,close,volume")
        for rec in reader:
            ts_raw = rec["timestamp"].strip()
            try:
                ts = int(ts_raw)
            except ValueError:
                ts = parse_date_to_epoch(ts_raw)
            rows.append(
                Row(
                    ts=ts,
                    open=float(rec["open"]),
                    high=float(rec["high"]),
                    low=float(rec["low"]),
                    close=float(rec["close"]),
                    volume=float(rec["volume"]),
                )
            )
    rows.sort(key=lambda r: r.ts)
    if len(rows) < 300:
        raise ValueError("Need at least 300 OHLCV rows")
    return rows


def fetch_yahoo_ohlcv(symbol: str, start: str, end: str, interval: str = "1d") -> list[Row]:
    p1 = parse_date_to_epoch(start)
    p2 = parse_date_to_epoch(end)
    query = urllib.parse.urlencode(
        {
            "period1": p1,
            "period2": p2,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        payload = json.loads(r.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise ValueError("Yahoo response missing chart result")

    node = result[0]
    ts_list = node.get("timestamp") or []
    quote = (node.get("indicators", {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    vols = quote.get("volume") or []

    rows: list[Row] = []
    n = min(len(ts_list), len(opens), len(highs), len(lows), len(closes), len(vols))
    for i in range(n):
        vals = (ts_list[i], opens[i], highs[i], lows[i], closes[i], vols[i])
        if any(v is None for v in vals):
            continue
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        if c <= 0 or h <= 0 or l <= 0 or o <= 0:
            continue
        rows.append(Row(ts=int(ts_list[i]), open=o, high=h, low=l, close=c, volume=float(vols[i])))

    if len(rows) < 300:
        raise ValueError(f"Not enough Yahoo rows for {symbol}: {len(rows)}")
    return rows


def pct_change(vals: list[float], lag: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    for i in range(lag, len(vals)):
        prev = vals[i - lag]
        out[i] = (vals[i] / prev - 1.0) if prev else None
    return out


def rolling_mean(vals: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= window:
            s -= vals[i - window]
        if i >= window - 1:
            out[i] = s / window
    return out


def rolling_std(vals: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    for i in range(window - 1, len(vals)):
        chunk = vals[i - window + 1 : i + 1]
        m = sum(chunk) / window
        out[i] = math.sqrt(sum((x - m) ** 2 for x in chunk) / window)
    return out


def rsi(closes: list[float], window: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    gains = [0.0] * len(closes)
    losses = [0.0] * len(closes)
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains[i] = max(d, 0.0)
        losses[i] = max(-d, 0.0)
    for i in range(window, len(closes)):
        avg_g = sum(gains[i - window + 1 : i + 1]) / window
        avg_l = sum(losses[i - window + 1 : i + 1]) / window
        if avg_l == 0:
            out[i] = 100.0
        else:
            rs = avg_g / avg_l
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def atr(high: list[float], low: list[float], close: list[float], window: int = 14) -> list[float | None]:
    tr = [0.0] * len(close)
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return rolling_mean(tr, window)


def build_dataset(rows: list[Row]) -> tuple[list[list[float]], list[float], list[int]]:
    close = [r.close for r in rows]
    high = [r.high for r in rows]
    low = [r.low for r in rows]
    open_ = [r.open for r in rows]
    vol = [r.volume for r in rows]

    ret1 = pct_change(close, 1)
    ret3 = pct_change(close, 3)
    ret5 = pct_change(close, 5)
    ret10 = pct_change(close, 10)
    ma5 = rolling_mean(close, 5)
    ma20 = rolling_mean(close, 20)
    ma50 = rolling_mean(close, 50)
    vol_std20 = rolling_std([x if x is not None else 0.0 for x in ret1], 20)
    vol_chg5 = pct_change(vol, 5)
    rsi14 = rsi(close, 14)
    atr14 = atr(high, low, close, 14)

    x: list[list[float]] = []
    y_ret: list[float] = []
    y_dir: list[int] = []

    for i in range(len(rows) - 1):
        vals = [ret1[i], ret3[i], ret5[i], ret10[i], ma5[i], ma20[i], ma50[i], vol_std20[i], vol_chg5[i], rsi14[i], atr14[i]]
        if any(v is None for v in vals):
            continue

        next_ret = close[i + 1] / close[i] - 1.0
        candle_body = (close[i] - open_[i]) / open_[i]
        hl_range = (high[i] - low[i]) / close[i]

        feats = [
            float(ret1[i]),
            float(ret3[i]),
            float(ret5[i]),
            float(ret10[i]),
            float(ma5[i] / close[i] - 1.0),
            float(ma20[i] / close[i] - 1.0),
            float(ma50[i] / close[i] - 1.0),
            float((ma5[i] - ma20[i]) / close[i]),
            float(vol_std20[i]),
            float(vol[i] / 1_000_000.0),
            float(vol_chg5[i]),
            float(rsi14[i] / 100.0),
            float(atr14[i] / close[i]),
            float(candle_body),
            float(hl_range),
            float(ret1[i]) * float(hl_range),
            float(ret3[i]) * float(vol_std20[i]),
        ]

        x.append(feats)
        y_ret.append(next_ret)
        y_dir.append(1 if next_ret >= 0 else 0)

    if len(x) < 220:
        raise ValueError("Not enough feature rows after engineering")
    return x, y_ret, y_dir


def standardize_fit(features: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(features[0])
    means = [sum(r[j] for r in features) / len(features) for j in range(n)]
    stds = []
    for j in range(n):
        var = sum((r[j] - means[j]) ** 2 for r in features) / len(features)
        s = math.sqrt(var)
        stds.append(s if s > 1e-12 else 1.0)
    return means, stds


def standardize_apply(features: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(r[j] - means[j]) / stds[j] for j in range(len(r))] for r in features]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_logistic_sgd(features: list[list[float]], labels: list[int], lr: float, epochs: int, l2: float) -> tuple[list[float], float]:
    w = [0.0] * len(features[0])
    b = 0.0
    for _ in range(epochs):
        for x, y in zip(features, labels):
            p = sigmoid(b + sum(wi * xi for wi, xi in zip(w, x)))
            e = p - y
            b -= lr * e
            for j in range(len(w)):
                w[j] -= lr * (e * x[j] + l2 * w[j])
    return w, b


def predict_many_proba(features: Iterable[list[float]], weights: list[float], bias: float) -> list[float]:
    return [sigmoid(bias + sum(w * x for w, x in zip(weights, row))) for row in features]


def predict_momentum_direction(features_raw: list[list[float]], lag_idx: int = 0) -> list[float]:
    return [0.70 if r[lag_idx] >= 0 else 0.30 for r in features_raw]


def directional_accuracy(y_true: list[int], probs: list[float], thr: float = 0.5) -> float:
    pred = [1 if p >= thr else 0 for p in probs]
    return sum(int(a == b) for a, b in zip(y_true, pred)) / len(y_true)


def thresholded_hit_rate(y_true: list[int], probs: list[float], thr: float) -> tuple[float, float]:
    idx = [i for i, p in enumerate(probs) if p >= thr or p <= (1.0 - thr)]
    if not idx:
        return 0.0, 0.0
    hit = sum(int((1 if probs[i] >= thr else 0) == y_true[i]) for i in idx) / len(idx)
    cov = len(idx) / len(y_true)
    return cov, hit


def tune_threshold(y_true: list[int], probs: list[float], min_cov: float = 0.20) -> tuple[float, float, float]:
    best_thr, best_hit, best_cov = 0.5, directional_accuracy(y_true, probs, 0.5), 1.0
    for s in range(50, 76):
        thr = s / 100.0
        cov, hit = thresholded_hit_rate(y_true, probs, thr)
        if cov < min_cov:
            continue
        if (0.8 * hit + 0.2 * cov) > (0.8 * best_hit + 0.2 * best_cov):
            best_thr, best_hit, best_cov = thr, hit, cov
    return best_thr, best_cov, best_hit


def walk_forward_directional_accuracy(x_raw: list[list[float]], y_dir: list[int], train_window: int = 500, test_window: int = 120) -> float:
    scores: list[float] = []
    i = 0
    while i + train_window + test_window <= len(x_raw):
        xtr = x_raw[i : i + train_window]
        ytr = y_dir[i : i + train_window]
        xte = x_raw[i + train_window : i + train_window + test_window]
        yte = y_dir[i + train_window : i + train_window + test_window]
        m, s = standardize_fit(xtr)
        xtr_s = standardize_apply(xtr, m, s)
        xte_s = standardize_apply(xte, m, s)
        w, b = fit_logistic_sgd(xtr_s, ytr, lr=0.02, epochs=260, l2=0.0006)
        p = predict_many_proba(xte_s, w, b)
        scores.append(directional_accuracy(yte, p, 0.5))
        i += test_window
    return sum(scores) / len(scores) if scores else 0.0


def select_best_strategy(x_val_raw: list[list[float]], x_val_std: list[list[float]], y_val: list[int], ensemble_models: list[dict]) -> tuple[str, list[float]]:
    p_logit = [sum(v) / len(v) for v in zip(*[predict_many_proba(x_val_std, m["weights"], m["bias"]) for m in ensemble_models])]
    p_mom1 = predict_momentum_direction(x_val_raw, 0)
    p_mom3 = predict_momentum_direction(x_val_raw, 1)
    p_blend = [0.65 * a + 0.35 * b for a, b in zip(p_logit, p_mom1)]

    cands = {
        "logistic_ensemble": p_logit,
        "momentum_ret1": p_mom1,
        "momentum_ret3": p_mom3,
        "blend_logit_mom1": p_blend,
    }
    best = max(cands.items(), key=lambda kv: directional_accuracy(y_val, kv[1], 0.5))
    return best[0], best[1]


def apply_strategy(name: str, x_raw: list[list[float]], x_std: list[list[float]], ensemble_models: list[dict]) -> list[float]:
    if name == "logistic_ensemble":
        return [sum(v) / len(v) for v in zip(*[predict_many_proba(x_std, m["weights"], m["bias"]) for m in ensemble_models])]
    if name == "momentum_ret1":
        return predict_momentum_direction(x_raw, 0)
    if name == "momentum_ret3":
        return predict_momentum_direction(x_raw, 1)
    if name == "blend_logit_mom1":
        p1 = apply_strategy("logistic_ensemble", x_raw, x_std, ensemble_models)
        p2 = predict_momentum_direction(x_raw, 0)
        return [0.65 * a + 0.35 * b for a, b in zip(p1, p2)]
    raise ValueError(f"Unknown strategy: {name}")


def get_rows(csv_path: Path | None, symbol: str, source: str, start: str, end: str) -> tuple[list[Row], str]:
    if csv_path:
        return load_csv(csv_path), f"csv:{csv_path}"

    if source == "yahoo":
        try:
            rows = fetch_yahoo_ohlcv(symbol=symbol, start=start, end=end, interval="1d")
            return rows, f"yahoo:{symbol}"
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"[WARN] Yahoo fetch failed ({exc}); fallback to synthetic data")

    return generate_synthetic_data(), "synthetic:fallback"


def train(csv_path: Path | None, symbol: str, source: str, start: str, end: str) -> dict:
    rows, data_source = get_rows(csv_path, symbol, source, start, end)
    x_raw, y_ret, y_dir = build_dataset(rows)

    n = len(x_raw)
    tr_end = int(n * 0.70)
    val_end = int(n * 0.85)

    x_train, x_val, x_test = x_raw[:tr_end], x_raw[tr_end:val_end], x_raw[val_end:]
    y_train, y_val, y_test = y_dir[:tr_end], y_dir[tr_end:val_end], y_dir[val_end:]

    means, stds = standardize_fit(x_train)
    x_train_s = standardize_apply(x_train, means, stds)
    x_val_s = standardize_apply(x_val, means, stds)
    x_test_s = standardize_apply(x_test, means, stds)

    grid = [
        {"lr": 0.015, "epochs": 320, "l2": 0.0002},
        {"lr": 0.020, "epochs": 280, "l2": 0.0005},
        {"lr": 0.025, "epochs": 240, "l2": 0.0010},
        {"lr": 0.030, "epochs": 200, "l2": 0.0012},
    ]

    scored = []
    for cfg in grid:
        w, b = fit_logistic_sgd(x_train_s, y_train, cfg["lr"], cfg["epochs"], cfg["l2"])
        pv = predict_many_proba(x_val_s, w, b)
        scored.append((directional_accuracy(y_val, pv, 0.5), cfg, w, b))
    scored.sort(key=lambda t: t[0], reverse=True)

    ens = [{"cfg": cfg, "weights": w, "bias": b} for _, cfg, w, b in scored[:3]]
    strat, p_val = select_best_strategy(x_val, x_val_s, y_val, ens)
    thr, val_cov, val_hit = tune_threshold(y_val, p_val, min_cov=0.25)

    p_test = apply_strategy(strat, x_test, x_test_s, ens)
    acc = directional_accuracy(y_test, p_test, 0.5)
    cov, hit = thresholded_hit_rate(y_test, p_test, thr)
    wf = walk_forward_directional_accuracy(x_raw, y_dir)

    model = {
        "task": "direction",
        "selected_strategy": strat,
        "tuned_threshold": thr,
        "ensemble_models": [{"weights": m["weights"], "bias": m["bias"], "cfg": m["cfg"]} for m in ens],
        "means": means,
        "stds": stds,
        "symbol": symbol,
        "source": source,
        "start": start,
        "end": end,
        "trained_at": int(time.time()),
    }

    metrics = {
        "task": "direction",
        "data_source": data_source,
        "symbol": symbol,
        "source": source,
        "split": "70/15/15",
        "selected_strategy": strat,
        "directional_accuracy": acc,
        "walk_forward_directional_accuracy": wf,
        "tuned_threshold": thr,
        "validation_threshold_coverage": val_cov,
        "validation_threshold_hit_rate": val_hit,
        "threshold_coverage": cov,
        "threshold_hit_rate": hit,
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "test_rows": len(x_test),
    }

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
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None, help="CSV path with timestamp,open,high,low,close,volume")
    p.add_argument("--symbol", default="AAPL", help="Ticker or crypto symbol for Yahoo (e.g., AAPL, BTC-USD)")
    p.add_argument("--source", choices=["yahoo"], default="yahoo")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = p.parse_args()
    train(args.csv, args.symbol, args.source, args.start, args.end)


if __name__ == "__main__":
    main()
