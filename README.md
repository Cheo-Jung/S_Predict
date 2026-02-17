# Real-Market OHLCV Direction Predictor (MVP)

This project is now **real-data-first**:
- Fetches daily OHLCV from Yahoo Finance over HTTP (no third-party packages).
- Supports CSV OHLCV input for your own exchange/broker data.
- Trains a direction predictor with time-aware split and validation-based strategy selection.

## Supported data inputs
1. **Yahoo OHLCV** (default)
2. **CSV** with columns:
   - `timestamp` (unix seconds or `YYYY-MM-DD`)
   - `open,high,low,close,volume`

## Core features used
- Returns: 1/3/5/10 day
- MA deviations: MA5, MA20, MA50
- Trend spread: MA5-MA20
- Volatility: rolling std of returns
- Volume change: 5-day
- RSI(14)
- ATR(14)
- Candle body and high-low range

## Train on real market data
```bash
python train_mvp.py --symbol AAPL --source yahoo --start 2016-01-01 --end 2026-01-01
```

For crypto:
```bash
python train_mvp.py --symbol BTC-USD --source yahoo --start 2016-01-01 --end 2026-01-01
```

## Train from CSV
```bash
python train_mvp.py --csv data/ohlcv.csv
```

## Predict latest signal
```bash
python predict_mvp.py --symbol AAPL --start 2016-01-01 --end 2026-01-01
```

## API
```bash
python api_mvp.py
```

Endpoints:
- `GET /health`
- `GET /predict?symbol=AAPL&start=2016-01-01&end=2026-01-01`

## Important trading note
This is still an MVP model. Do not deploy with meaningful capital without:
- transaction cost + slippage modeling,
- out-of-sample robustness checks across market regimes,
- strict risk management and position sizing,
- paper-trading validation.
