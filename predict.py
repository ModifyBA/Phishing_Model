# predict.py
import os
import pickle
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from FE import extract_features


# =========================================================
# PATHS / CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lbgm_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
DB_PATH = BASE_DIR / "phishing.db"

PHISHING_THRESHOLD = float(os.getenv("PHISH_THRESHOLD", "0.90"))
SUSPICIOUS_THRESHOLD = float(os.getenv("SUSPICIOUS_THRESHOLD", "0.65"))

TRUSTED_DOMAINS = {
    d.strip().lower()
    for d in os.getenv(
        "TRUSTED_DOMAINS",
        "google.com,github.com,microsoft.com,apple.com,openai.com"
    ).split(",")
    if d.strip()
}

DISCORD_INVITE_VERDICT = os.getenv("DISCORD_INVITE_VERDICT", "suspicious").strip().lower()
if DISCORD_INVITE_VERDICT not in {"phishing", "suspicious"}:
    DISCORD_INVITE_VERDICT = "suspicious"

_lock = threading.Lock()
_cached_model: Any = None
_cached_cols: List[str] | None = None


# =========================================================
# URL HELPERS
# =========================================================
def _ensure_scheme(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    return u


def normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        raise ValueError("URL is empty")
    if "://" not in u:
        u = "http://" + u
    return u


def _host_path(url: str) -> Tuple[str, str]:
    u = _ensure_scheme(url)
    p = urlparse(u)
    host = (p.hostname or "").lower().rstrip(".")
    path = (p.path or "").lower()
    return host, path


def is_trusted_host(url: str) -> bool:
    host, _ = _host_path(url)
    if not host:
        return False

    for domain in TRUSTED_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True
    return False


def is_discord_invite(url: str) -> bool:
    host, path = _host_path(url)
    if not host:
        return False

    if host == "discord.gg" and path.strip("/") != "":
        return True

    if (host == "discord.com" or host.endswith(".discord.com")) and path.startswith("/invite/"):
        return True

    return False


# =========================================================
# ARTIFACT LOADING
# =========================================================
def load_train_cols() -> List[str]:
    global _cached_cols

    with _lock:
        if _cached_cols is not None:
            return _cached_cols

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"Missing feature_names.pkl: {FEATURES_PATH}")

        with open(FEATURES_PATH, "rb") as f:
            cols = pickle.load(f)

        if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
            raise ValueError("feature_names.pkl must be a non-empty list[str]")

        _cached_cols = cols
        return cols


def load_bundle() -> Dict[str, Any]:
    global _cached_model

    with _lock:
        if _cached_model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
            _cached_model = joblib.load(MODEL_PATH)

    return {
        "model": _cached_model,
        "phishing_threshold": PHISHING_THRESHOLD,
        "suspicious_threshold": SUSPICIOUS_THRESHOLD,
    }


# =========================================================
# FEATURE BUILDING
# =========================================================
def _make_X(url: str, train_cols: List[str]) -> pd.DataFrame:
    feats = extract_features(url)

    if not isinstance(feats, dict):
        raise ValueError("extract_features(url) must return a dict")

    X = pd.DataFrame([feats])

    if X.empty:
        raise ValueError("Feature extraction returned no features")

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)
    X = X.reindex(columns=train_cols, fill_value=0)

    if X.empty or X.shape[1] == 0:
        raise ValueError("Feature matrix is empty after column alignment")

    return X


# =========================================================
# DATABASE LOGGING
# =========================================================
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        prediction_score REAL,
        classification TEXT,
        timestamp TEXT,
        model_version TEXT,
        source TEXT
    )
    """)

    conn.commit()
    conn.close()


def log_prediction(url: str, score: float, classification: str, source: str = "extension") -> None:
    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions
    (url, prediction_score, classification, timestamp, model_version, source)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        url,
        float(score),
        classification,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lightgbm_v1",
        source,
    ))

    conn.commit()
    conn.close()


# =========================================================
# PREDICTION LOGIC
# =========================================================
def predict_with_bundle(
    url: str,
    bundle: Dict[str, Any],
    train_cols: List[str],
) -> Tuple[str, float, float, float, str]:
    normalized_url = normalize_url(url)

    phishing_threshold = float(bundle.get("phishing_threshold", PHISHING_THRESHOLD))
    suspicious_threshold = float(bundle.get("suspicious_threshold", SUSPICIOUS_THRESHOLD))

    # Trusted domains bypass
    if is_trusted_host(normalized_url):
        return "legitimate", 0.0, phishing_threshold, suspicious_threshold, "rule"

    # Rule for Discord invites
    if is_discord_invite(normalized_url):
        return DISCORD_INVITE_VERDICT, 0.99, phishing_threshold, suspicious_threshold, "rule"

    # ML prediction
    model = bundle["model"]
    X = _make_X(normalized_url, train_cols)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        prob = float(model.predict(X)[0])

    if prob >= phishing_threshold:
        verdict = "phishing"
    elif prob >= suspicious_threshold:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    log_prediction(normalized_url, prob, verdict)
    return verdict, prob, phishing_threshold, suspicious_threshold, "model"


def predict_url(url: str) -> Dict[str, Any]:
    try:
        train_cols = load_train_cols()
        bundle = load_bundle()

        verdict, prob, phishing_threshold, suspicious_threshold, decision_source = predict_with_bundle(
            url=url,
            bundle=bundle,
            train_cols=train_cols,
        )

        return {
            "success": True,
            "url": normalize_url(url),
            "classification": verdict,
            "prediction_score": round(prob, 6),
            "phishing_threshold": phishing_threshold,
            "suspicious_threshold": suspicious_threshold,
            "should_warn": verdict in {"suspicious", "phishing"},
            "should_block": verdict == "phishing",
            "decision_source": decision_source,
        }

    except Exception as e:
        return {
            "success": False,
            "url": url,
            "classification": "error",
            "prediction_score": None,
            "phishing_threshold": PHISHING_THRESHOLD,
            "suspicious_threshold": SUSPICIOUS_THRESHOLD,
            "should_warn": False,
            "should_block": False,
            "decision_source": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    test_urls = [
        "google.com",
        "discord.gg/example",
        "paypal-login-secure.ru/verify/account",
        "github.com",
        "bit.ly/free-prize",
        "https://mass.gov-uxud.cfd/rmv",
        "http://allegro.42837.cfd/"
    ]

    for u in test_urls:
        print(predict_url(u))