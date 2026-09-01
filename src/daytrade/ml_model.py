"""Trains a small ensemble -- logistic regression, random forest, gradient
boosting -- to predict whether a setup is more likely to hit its
take-profit or its stop-loss first, using the same ~60-indicator feature
set the rule-based engine votes on (features.py) plus Fibonacci/ATR
context.

This is a *second opinion*, not a replacement for the rule-based signal.
autotrade.py only buys when both the rule-based STRONG BUY fires AND a
majority of these three models agree the trade has better than even odds.
Three different model families (linear, bagged trees, boosted trees) vote
together specifically so one model's blind spot doesn't decide alone.

Labels come from the "triple barrier" method: for each historical bar,
compute the ATR-based stop-loss/take-profit exactly as risk.py would at
that moment, then look forward up to `HORIZON` bars -- label 1 if price
hits the take-profit before the stop-loss, 0 if the stop is hit first,
and drop the bar if neither is hit in time (an unresolved trade is not
useful supervision, and keeping it would bias the model toward chop).

Usage:
    python train_models.py                  Train on the default universe
    python train_models.py --symbols AAPL,MSFT,NVDA --period 60d
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from daytrade import indicators, risk
from daytrade.features import FEATURE_COLUMNS, build_features

MODEL_DIR = Path("data/models")
HORIZON = 12    # bars forward to watch for a barrier hit (12 * 5m = 1 trading hour)
MIN_BARS = 80   # warmup room for indicators before the first sample
STRIDE = 3      # skip bars between samples so near-duplicate consecutive rows don't dominate


def _download(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=str.title).dropna()
    return df


def _label_bar(df: pd.DataFrame, i: int, horizon: int = HORIZON) -> int | None:
    """1 if the ATR take-profit is hit before the ATR stop-loss within
    `horizon` bars forward of bar i, 0 if the stop is hit first, None if
    neither is hit (caller should drop these)."""
    entry_price = float(df["Close"].iloc[i])
    plan = risk.compute_stops(df.iloc[: i + 1], entry_price)
    future = df.iloc[i + 1 : i + 1 + horizon]
    if future.empty:
        return None

    for _, bar in future.iterrows():
        hit_take = bar["High"] >= plan.take_price
        hit_stop = bar["Low"] <= plan.stop_price
        if hit_take and hit_stop:
            return 0  # can't see intra-bar order -- assume the worse outcome
        if hit_take:
            return 1
        if hit_stop:
            return 0
    return None


def build_training_set(symbols: list[str], period: str = "60d", interval: str = "5m",
                        verbose: bool = True) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        df = _download(symbol, period, interval)
        if df is None or len(df) < MIN_BARS + HORIZON:
            if verbose:
                print(f"  [{symbol}] skipped (no/insufficient data)")
            continue

        signals = dict(indicators.signal_series(df))
        added = 0
        for i in range(MIN_BARS, len(df) - HORIZON, STRIDE):
            if i not in signals:
                continue
            label = _label_bar(df, i)
            if label is None:
                continue
            feats = build_features(signals[i])
            feats["label"] = label
            feats["symbol"] = symbol
            rows.append(feats)
            added += 1
        if verbose:
            print(f"  [{symbol}] {len(df)} bars -> {added} labeled samples")

    return pd.DataFrame(rows)


def train(training_df: pd.DataFrame) -> dict:
    """Per-symbol chronological 80/20 split, then concatenated -- keeps
    the test set strictly after what each model trained on. A random
    row-level split would leak future bars into training since adjacent
    rows are highly correlated."""
    train_parts, test_parts = [], []
    for _, g in training_df.groupby("symbol"):
        cut = int(len(g) * 0.8)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    train_df = pd.concat(train_parts) if train_parts else training_df.iloc[0:0]
    test_df = pd.concat(test_parts) if test_parts else training_df.iloc[0:0]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test) if len(X_test) else X_test

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=25,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        ),
    }

    metrics = {}
    for name, model in models.items():
        xtr = X_train_s if name == "logistic_regression" else X_train
        xte = X_test_s if name == "logistic_regression" else X_test
        model.fit(xtr, y_train)
        if len(X_test):
            preds = model.predict(xte)
            proba = model.predict_proba(xte)[:, 1]
            metrics[name] = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "precision": round(precision_score(y_test, preds, zero_division=0), 4),
                "auc": round(roc_auc_score(y_test, proba), 4) if len(set(y_test)) > 1 else None,
                "n_test": int(len(y_test)),
            }

    return {
        "models": models, "scaler": scaler, "metrics": metrics,
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "base_rate": round(float(y_train.mean()), 4) if len(y_train) else None,
    }


def save(trained: dict, model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained["models"].items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    joblib.dump(trained["scaler"], model_dir / "scaler.joblib")
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "metrics": trained["metrics"],
        "n_train": trained["n_train"],
        "n_test": trained["n_test"],
        "base_rate": trained["base_rate"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_ensemble(model_dir: Path = MODEL_DIR) -> dict | None:
    """None if no trained models are on disk -- callers should treat that
    as "ML gating unavailable" and fall back to rule-based-only."""
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    models = {}
    for path in model_dir.glob("*.joblib"):
        if path.stem == "scaler":
            continue
        models[path.stem] = joblib.load(path)
    if not models:
        return None
    scaler_path = model_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return {"models": models, "scaler": scaler, "feature_columns": meta["feature_columns"], "metadata": meta}


def predict_ensemble(ensemble: dict, feats: dict) -> dict:
    """Runs every model in the ensemble on one feature row. `agree` is
    what autotrade.py gates entries on: a majority of models giving the
    trade better than 50/50 odds of hitting take-profit before stop-loss."""
    cols = ensemble["feature_columns"]
    x = pd.DataFrame([{col: feats.get(col, 0.0) for col in cols}])[cols]
    x_scaled = ensemble["scaler"].transform(x) if ensemble["scaler"] is not None else x

    probs = {}
    for name, model in ensemble["models"].items():
        xi = x_scaled if name == "logistic_regression" else x
        probs[name] = float(model.predict_proba(xi)[0, 1])

    votes = sum(1 for p in probs.values() if p >= 0.5)
    return {
        "probabilities": probs,
        "avg_probability": round(sum(probs.values()) / len(probs), 4) if probs else None,
        "votes_for": votes,
        "n_models": len(probs),
        "agree": votes >= (len(probs) // 2 + 1),
    }
