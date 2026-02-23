# Stock Predictor MVP Stack (Offline Runnable)

This MVP is intentionally dependency-light so it can run in restricted environments.

## Components
- `train_mvp.py`: trains either
  - a **direction classifier** (logistic SGD, default), or
  - a **return regressor** (linear SGD)
- `predict_mvp.py`: loads artifacts and predicts the latest direction/return.
- `api_mvp.py`: small HTTP server with `/health`, `/predict`, `/meta`, and a built-in web GUI at `/`.

## Data source
- Optional: pass your own CSV with columns: `close,volume`.
- Default: deterministic synthetic price/volume data is generated.

## Why this version improves directional accuracy
- Direction task now optimizes a **classification objective** directly.
- Features expanded (multi-horizon returns, trend spread, vol features, volume change).
- Includes thresholded confidence reporting (`threshold_coverage`, `threshold_hit_rate`).
- Includes walk-forward directional validation.

## Train + Evaluate (direction mode)
```bash
python train_mvp.py --task direction --threshold 0.55
```

## Optional regression mode
```bash
python train_mvp.py --task return
```

## Predict
```bash
python predict_mvp.py
```

## Serve API
```bash
python api_mvp.py
```

Then open:
- `http://127.0.0.1:8000/` (GUI dashboard)
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/predict`
- `http://127.0.0.1:8000/meta`
