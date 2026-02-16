# MVP Execution & Accuracy Evaluation

The updated MVP source code was executed end-to-end in this environment.

## Commands run
- `python -m py_compile train_mvp.py predict_mvp.py api_mvp.py`
- `python train_mvp.py --task direction --threshold 0.55`
- `python train_mvp.py --task return`
- `python predict_mvp.py`
- API smoke test via `python api_mvp.py` with `curl /health` and `curl /predict`

## Direction-task evaluation (primary)
Metrics reported by `python train_mvp.py --task direction --threshold 0.55`:

- Directional accuracy: `0.550676` (55.07%)
- Walk-forward directional accuracy: `0.532500` (53.25%)
- Threshold: `0.55`
- Threshold coverage: `0.787162` (78.72%)
- Threshold hit rate: `0.562232` (56.22%)
- Train rows: `1184`
- Test rows: `296`
- Data source: `synthetic`

## Regression-task comparison
Metrics reported by `python train_mvp.py --task return`:

- MAE: `0.011568`
- RMSE: `0.014696`
- Directional accuracy (sign-based): `0.452703`

## Interpretation
- Compared to the prior baseline (~51% directional), the direction-optimized mode improves hit rate in this MVP synthetic setup.
- Thresholded evaluation is useful to trade only higher-confidence predictions.
- For production relevance, run the same pipeline on real OHLCV CSV data.
