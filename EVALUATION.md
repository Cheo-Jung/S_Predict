# Evaluation (Real-Data-Capable Version)

## What changed
- Added real OHLCV ingestion from Yahoo (`query1.finance.yahoo.com`) and CSV support.
- Added OHLCV technical features (RSI/ATR/MA/volatility/candle structure).
- Kept time-aware train/validation/test split and strategy selection.

## Commands run
- `python -m py_compile train_mvp.py predict_mvp.py api_mvp.py`
- `python train_mvp.py --symbol AAPL --source yahoo --start 2016-01-01 --end 2026-01-01`
- `python predict_mvp.py --symbol AAPL --start 2016-01-01 --end 2026-01-01`
- `python api_mvp.py` then `curl /health` and `curl /predict?symbol=AAPL`

## Notes
- If Yahoo access fails in restricted environments, code falls back to synthetic data and prints a warning.
- In normal networked environments this should train on real OHLCV for stocks/crypto symbols supported by Yahoo (e.g., `AAPL`, `BTC-USD`).
