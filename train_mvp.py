from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "meta.joblib"


def load_data(symbol: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    return df.rename(columns=str.lower)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_5"] = out["close"].pct_change(5)
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["target"] = out["close"].shift(-1) / out["close"] - 1
    return out.dropna()


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut], df.iloc[cut:]


def train(symbol: str = "AAPL") -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data(symbol=symbol)
    feat = make_features(raw)

    feature_cols = ["ret_1", "ret_5", "ma_5", "ma_20", "vol_20", "volume"]
    train_df, test_df = time_split(feat)

    x_train = train_df[feature_cols]
    y_train = train_df["target"]
    x_test = test_df[feature_cols]
    y_test = test_df["target"]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)

    print(f"[MVP] Symbol={symbol}, Test MAE={mae:.6f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"symbol": symbol, "feature_cols": feature_cols, "mae": mae}, META_PATH)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved meta  -> {META_PATH}")


if __name__ == "__main__":
    train("AAPL")
