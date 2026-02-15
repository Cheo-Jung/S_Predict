# Stock Predictor MVP Stack

This repository contains a minimal, runnable MVP for stock next-day return prediction.

## Stack
- Data: `yfinance`
- Feature engineering: `pandas`
- Model: `LightGBM`
- Serving API: `FastAPI`
- Serialization: `joblib`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model
```bash
python train_mvp.py
```

Artifacts are saved to `artifacts/model.joblib` and `artifacts/meta.joblib`.

## Run single prediction
```bash
python predict_mvp.py
```

## Run API
```bash
uvicorn api_mvp:app --reload --port 8000
```

### Endpoints
- `GET /health`
- `POST /predict` with body:
```json
{
  "symbol": "AAPL"
}
```
