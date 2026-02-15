from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/meta.joblib")


def latest_features(symbol: str, feature_cols: list[str]) -> pd.DataFrame:
    df = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}")

    df = df.rename(columns=str.lower)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough data to build features")

    return df[feature_cols].tail(1)


def predict(symbol: str = "AAPL") -> None:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artifacts missing. Run `python train_mvp.py` first.")

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    x = latest_features(symbol, meta["feature_cols"])
    yhat = float(model.predict(x)[0])
    print(f"Predicted next-day return for {symbol}: {yhat:.6%}")


if __name__ == "__main__":
    predict("AAPL")
