from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/meta.joblib")

app = FastAPI(title="Stock Predictor MVP")


class PredictRequest(BaseModel):
    symbol: str = "AAPL"


def build_latest_features(symbol: str, feature_cols: list[str]) -> pd.DataFrame:
    df = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No price data")

    df = df.rename(columns=str.lower)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough history for features")

    return df[feature_cols].tail(1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifacts missing. Run training first.")

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    try:
        x = build_latest_features(req.symbol, meta["feature_cols"])
        pred = float(model.predict(x)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "symbol": req.symbol,
        "predicted_next_day_return": pred,
        "model_mae": meta.get("mae"),
        "feature_cols": meta["feature_cols"],
    }
