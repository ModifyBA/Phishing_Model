# predict.py
import os
import pickle
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import json
from openai import OpenAI

def load_openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()

    local_key_path = Path(__file__).resolve().parent / "openai_api_key.txt"
    if local_key_path.exists():
        return local_key_path.read_text(encoding="utf-8").strip()

    return None


client: OpenAI | None = None


def get_openai_client() -> OpenAI | None:
    global client

    if client is not None:
        return client

    api_key = load_openai_api_key()
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    return client
import joblib
import numpy as np
import pandas as pd
import socket
import ssl
import time
from datetime import timezone
from Feature_Extract import extract_features


# =========================================================
# PATHS / CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lbgm_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
DB_PATH = BASE_DIR / "phishing.db"
THRESHOLD_PATH = BASE_DIR / "threshold.json"

PHISHING_THRESHOLD = float(os.getenv("PHISH_THRESHOLD", "0.85"))
SUSPICIOUS_THRESHOLD = float(os.getenv("SUSPICIOUS_THRESHOLD", "0.65"))
ENABLE_AI_FEEDBACK = os.getenv("ENABLE_AI_FEEDBACK", "false").strip().lower() == "true"

TRUSTED_DOMAINS = {
    d.strip().lower()
    for d in os.getenv(
        "TRUSTED_DOMAINS",
        "apple.com,bbc.com,cisa.gov,cloudflare.com,facebook.com,ftc.gov,github.com,instagram.com,irs.gov,linkedin.com,live.com,microsoft.com,microsoftonline.com,mit.edu,mozilla.org,nytimes.com,openai.com,paypal.com,stanford.edu,wikipedia.org"
    ).split(",")
    if d.strip()
}

SAFE_IP_HOSTS = {"1.1.1.1"}

URL_SHORTENER_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "t.co",
    "tinyurl.com",
}

ABUSED_HOSTING_DOMAINS = {
    "sites.google.com",
    "docs.google.com",
    "github.io",
    "githubusercontent.com",
    "raw.githubusercontent.com",
    "storage.googleapis.com",
    "s3.amazonaws.com",
    "pastebin.com",
    "medium.com",
}

BRAND_LOOKALIKE_MARKERS = {
    "g00gle",
    "goog1e",
    "paypai",
    "paypa1",
    "micros0ft",
    "rnicrosoft",
    "amaz0n",
    "netfiix",
    "disc0rd",
    "faceboook",
    "steamcommunnity",
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
    parsed = urlparse(u)
    host = (parsed.hostname or "").lower().rstrip(".")

    https_preferred_hosts = {
        "accounts.google.com",
        "appleid.apple.com",
        "github.com",
        "google.com",
        "login.microsoftonline.com",
        "support.apple.com",
        "www.amazon.com",
        "www.bbc.com",
        "www.github.com",
        "www.google.com",
        "www.linkedin.com",
        "x.com",
    }

    if parsed.scheme == "http" and host in https_preferred_hosts:
        u = "https://" + u[len("http://"):]

    return u


def _host_path(url: str) -> Tuple[str, str]:
    u = _ensure_scheme(url)
    p = urlparse(u)
    host = (p.hostname or "").lower().rstrip(".")
    path = (p.path or "").lower()
    return host, path


def is_valid_public_url(url: str) -> bool:
    raw = (url or "").strip()
    if not raw or any(ch.isspace() for ch in raw):
        return False

    parsed = urlparse(_ensure_scheme(raw))
    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False

    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return False

    return True


def _is_host_in(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def is_trusted_host(url: str) -> bool:
    host, _ = _host_path(url)
    if not host:
        return False

    for domain in TRUSTED_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True
        
    if host.endswith(".gov") or host.endswith(".edu"):
        return True
    
    return False


def is_trusted_canonical_host(url: str) -> bool:
    host, _ = _host_path(url)
    if not host:
        return False

    canonical_hosts = {
        "accounts.google.com",
        "appleid.apple.com",
        "github.com",
        "login.microsoftonline.com",
        "support.apple.com",
        "www.amazon.com",
        "www.bbc.com",
        "www.github.com",
        "www.google.com",
        "www.linkedin.com",
        "x.com",
    }

    if host in canonical_hosts:
        return True

    for domain in TRUSTED_DOMAINS:
        if host == domain or host == "www." + domain:
            return True

    return host.endswith(".gov") or host.endswith(".edu")


def is_safe_ip_host(url: str) -> bool:
    host, _ = _host_path(url)
    return host in SAFE_IP_HOSTS


def is_url_shortener_host(url: str) -> bool:
    host, _ = _host_path(url)
    return bool(host and _is_host_in(host, URL_SHORTENER_DOMAINS))


def is_abused_hosting_host(url: str) -> bool:
    host, _ = _host_path(url)
    if not host:
        return False

    for domain in ABUSED_HOSTING_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True

    return False


def has_brand_lookalike_marker(url: str) -> bool:
    host, _ = _host_path(url)
    if not host or is_trusted_host(url):
        return False

    compact_host = host.replace("-", "").replace(".", "")
    return any(marker.replace("-", "") in compact_host for marker in BRAND_LOOKALIKE_MARKERS)


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
# CERTIFICATE CHECKING (ASYNC-LIKE / CONDITIONAL)
# =========================================================
CERT_CACHE = {}
CERT_CACHE_TTL = 60 * 60  # 1 hour


def _get_hostname(url: str) -> str | None:
    try:
        p = urlparse(_ensure_scheme(url))
        return (p.hostname or "").lower()
    except Exception:
        return None


def _check_certificate(url: str, timeout: float = 2.0) -> dict:
    host = _get_hostname(url)

    if not host:
        return {"checked": False, "error": "invalid_host"}

    if not url.lower().startswith("https://"):
        return {"checked": False, "error": "not_https"}

    context = ssl.create_default_context()

    try:
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()

        not_after = cert.get("notAfter")
        expired = None

        if not_after:
            expiry = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
            expired = expiry < datetime.now(timezone.utc)

        return {
            "checked": True,
            "cert_valid": True,
            "cert_expired": expired,
            "error": None
        }

    except ssl.SSLCertVerificationError:
        return {
            "checked": True,
            "cert_valid": False,
            "cert_expired": None,
            "error": "ssl_verification_failed"
        }

    except Exception:
        return {
            "checked": True,
            "cert_valid": None,
            "cert_expired": None,
            "error": "connection_failed"
        }


def get_cert_result(url: str) -> dict:
    host = _get_hostname(url)
    if not host:
        return _check_certificate(url)

    now = time.time()

    if host in CERT_CACHE:
        cached = CERT_CACHE[host]
        if now - cached["time"] < CERT_CACHE_TTL:
            return cached["data"]

    result = _check_certificate(url)

    CERT_CACHE[host] = {
        "time": now,
        "data": result
    }

    return result
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

def load_thresholds() -> tuple[float, float]:
    if not THRESHOLD_PATH.exists():
        return PHISHING_THRESHOLD, SUSPICIOUS_THRESHOLD

    with open(THRESHOLD_PATH, "r") as f:
        data = json.load(f)

    phishing_threshold = float(
        data.get("block_threshold", data.get("threshold", PHISHING_THRESHOLD))
    )
    suspicious_threshold = float(
        data.get("review_threshold", SUSPICIOUS_THRESHOLD)
    )

    return phishing_threshold, suspicious_threshold

def load_bundle() -> Dict[str, Any]:
    global _cached_model

    with _lock:
        if _cached_model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
            _cached_model = joblib.load(MODEL_PATH)

    phishing_threshold, suspicious_threshold = load_thresholds()

    return {
        "model": _cached_model,
        "phishing_threshold": phishing_threshold,
        "suspicious_threshold": suspicious_threshold,
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

def build_human_signals(
    url: str,
    verdict: str,
    prob: float,
    cert_info: Dict[str, Any] | None,
    decision_source: str,
) -> List[str]:
    signals: List[str] = []

    normalized_url = normalize_url(url)
    host, path = _host_path(normalized_url)

    suspicious_keywords = [
        "login", "signin", "verify", "secure", "update", "account",
        "confirm", "password", "bank", "wallet", "reset"
    ]
    found_keywords = [kw for kw in suspicious_keywords if kw in normalized_url.lower()]

    if not is_trusted_host(normalized_url):
        signals.append("The domain is not recognized as one of the trusted domains.")

    if found_keywords:
        shown = ", ".join(found_keywords[:3])
        signals.append(f"The URL contains suspicious keywords such as {shown}.")

    if is_discord_invite(normalized_url):
        signals.append("This is a Discord invite link, which can sometimes be abused in scams.")

    if is_url_shortener_host(normalized_url):
        signals.append("This URL uses a link shortener, so the final destination is hidden.")

    if is_abused_hosting_host(normalized_url):
        signals.append("This URL uses a public hosting platform where the path should be reviewed carefully.")

    if has_brand_lookalike_marker(normalized_url):
        signals.append("The domain resembles a well-known brand but is not the official domain.")

    if host and host.replace(".", "").isdigit() and not is_safe_ip_host(normalized_url):
        signals.append("The URL appears to use an IP-style host instead of a normal domain name.")

    if len(normalized_url) > 75:
        signals.append("The URL is unusually long, which can be used to hide misleading details.")

    if cert_info:
        if cert_info.get("cert_valid") is False:
            signals.append("The website has an invalid SSL certificate.")
        elif cert_info.get("cert_expired") is True:
            signals.append("The website’s SSL certificate appears to be expired.")
        elif cert_info.get("error") == "connection_failed":
            signals.append("The certificate check could not fully verify the secure connection.")

    if decision_source == "rule":
        signals.append("This result came from a rule-based safety check.")
    elif decision_source == "model+cert_rule":
        signals.append("The final result combined the machine learning score with certificate warnings.")
    elif decision_source == "model+cert_softener":
        signals.append("The certificate looked valid, which reduced the severity of the original model result.")
    elif decision_source == "model+trusted_softener":
        signals.append("The domain is commonly trusted, so the model result was softened.")
    elif decision_source == "model+lookalike_rule":
        signals.append("A brand-lookalike rule increased the severity of this result.")
    elif decision_source == "model+shortener_rule":
        signals.append("A link-shortener rule increased the severity of this result.")
    elif decision_source == "model":
        if prob >= 0.90:
            signals.append("The machine learning model detected strong phishing-related patterns.")
        elif prob >= 0.65:
            signals.append("The machine learning model detected several suspicious patterns.")
        else:
            signals.append("The machine learning model did not detect strong phishing patterns.")

    if verdict == "legitimate" and not signals:
        signals.append("The site matched expected patterns for a legitimate website.")

    # Keep only the strongest few so the AI explanation stays grounded and concise
    deduped: List[str] = []
    for s in signals:
        if s not in deduped:
            deduped.append(s)

    return deduped[:4]


def fallback_ai_feedback(classification: str, signals: List[str]) -> str:
    if signals:
        joined = " ".join(signals[:2])
    else:
        joined = "The system detected patterns that influenced this result."

    if classification == "phishing":
        return f"This website was flagged as phishing. {joined} Avoid entering passwords or personal information unless you can verify the site independently."
    if classification == "suspicious":
        return f"This website was marked as suspicious. {joined} Use caution before signing in or submitting sensitive information."
    return f"This website appears legitimate. {joined}"


# =========================================================
# GENAI FEEDBACK
# =========================================================
def generate_ai_feedback(
    url: str,
    classification: str,
    prediction_score: float,
    top_signals: List[str],
    trusted_domain_match: bool = False,
    ssl_status: str | None = None,
    decision_source: str | None = None,
) -> str:
    """
    Generates a short, user-friendly explanation grounded only in the
    signals supplied by the detection pipeline.
    """

    signal_summary = {
        "url": url,
        "classification": classification,
        "prediction_score": round(float(prediction_score), 4),
        "top_signals": top_signals,
        "trusted_domain_match": trusted_domain_match,
        "ssl_status": ssl_status,
        "decision_source": decision_source,
    }
    system_prompt = """
You are a cybersecurity assistant explaining phishing-detection results to normal users.

Rules:
1. Only use the evidence provided.
2. Do not invent reasons, threats, domains, or technical findings.
3. Keep the explanation to 2-4 short sentences.
4. Use simple, non-technical language.
5. Mention the strongest 2-3 signals only.
6. If classification is phishing, be clear and confident without sounding dramatic.
7. If classification is suspicious, clearly explain uncertainty.
8. If classification is legitimate, briefly explain why it appears safe.
9. Do not mention internal model names, SHAP, feature importance, or backend implementation.
10. End with a short safety tip for phishing or suspicious results.
Return only the explanation text.
"""

    user_prompt = f"""
Explain this website classification for a normal user.

Detection data:
{json.dumps(signal_summary, indent=2)}
"""

    openai_client = get_openai_client()
    if openai_client is None:
        return fallback_ai_feedback(classification, top_signals)

    response = openai_client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = getattr(response, "output_text", "") or ""
    text = text.strip()

    if not text:
        return fallback_ai_feedback(classification, top_signals)

    return text


def safe_generate_ai_feedback(
    url: str,
    classification: str,
    prediction_score: float,
    top_signals: List[str],
    trusted_domain_match: bool = False,
    ssl_status: str | None = None,
    decision_source: str | None = None,
) -> str:
    if not ENABLE_AI_FEEDBACK:
        return fallback_ai_feedback(classification, top_signals)

    try:
        return generate_ai_feedback(
            url=url,
            classification=classification,
            prediction_score=prediction_score,
            top_signals=top_signals,
            trusted_domain_match=trusted_domain_match,
            ssl_status=ssl_status,
            decision_source=decision_source,
        )
    except Exception:
        return fallback_ai_feedback(classification, top_signals)


# =========================================================
# PREDICTION LOGIC
# =========================================================
def predict_with_bundle(
    url: str,
    bundle: Dict[str, Any],
    train_cols: List[str],
) -> Tuple[str, float, float, float, str, Dict[str, Any] | None]:
    normalized_url = normalize_url(url)

    phishing_threshold = float(bundle.get("phishing_threshold", PHISHING_THRESHOLD))
    suspicious_threshold = float(bundle.get("suspicious_threshold", SUSPICIOUS_THRESHOLD))

    if not is_valid_public_url(normalized_url):
        return "invalid_url", 0.0, phishing_threshold, suspicious_threshold, "rule", None

    # Rule for Discord invites
    if is_discord_invite(normalized_url):
        return DISCORD_INVITE_VERDICT, 0.99, phishing_threshold, suspicious_threshold, "rule", None

    
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

    decision_source = "model"
    cert_info = None
    trusted = is_trusted_host(normalized_url)
    trusted_canonical = is_trusted_canonical_host(normalized_url)
    safe_ip = is_safe_ip_host(normalized_url)
    shortener = is_url_shortener_host(normalized_url)
    abused_hosting = is_abused_hosting_host(normalized_url)
    lookalike = has_brand_lookalike_marker(normalized_url)

    if lookalike and verdict == "legitimate":
        verdict = "suspicious"
        decision_source = "model+lookalike_rule"

    if safe_ip:
        verdict = "legitimate"
        decision_source = "rule"

    if shortener and verdict == "legitimate" and prob >= 0.30:
        verdict = "suspicious"
        decision_source = "model+shortener_rule"

    if trusted_canonical and not abused_hosting and not lookalike:
        if verdict in {"phishing", "suspicious"}:
            verdict = "legitimate"
            decision_source = "model+trusted_softener"
    elif trusted and not abused_hosting and not lookalike:
        if verdict == "phishing" and prob < 0.90:
            verdict = "suspicious"
            decision_source = "model+trusted_softener"
        elif verdict == "suspicious" and prob < 0.80:
            verdict = "legitimate"
            decision_source = "model+trusted_softener"

    # =========================================================
    # CERT CHECK (ONLY WHEN NEEDED)
    # =========================================================
    should_check_cert = (
    normalized_url.startswith("https://") and
    (
        verdict != "phishing" or
        prob > 0.4
    )
    )

    if should_check_cert:
        cert_info = get_cert_result(normalized_url)

        if cert_info.get("checked"):
            # Invalid cert → phishing
            if cert_info.get("cert_valid") is False:
                if prob >= phishing_threshold or lookalike:
                    verdict = "phishing"
                else:
                    verdict = "suspicious"
                decision_source = "model+cert_rule"

            # Expired cert → suspicious (only downgrade safe sites)
            elif cert_info.get("cert_expired") is True and verdict == "legitimate":
                verdict = "suspicious"
                decision_source = "model+cert_rule"

            elif (
                cert_info.get("cert_valid") is True
                and verdict == "phishing"
                and trusted_canonical
                and not abused_hosting
                and prob < 0.995
            ):
                verdict = "suspicious"
                decision_source = "model+cert_softener"
            elif (
                verdict == "phishing"
                and prob < 0.92
                and trusted_canonical
            ):
                verdict = "suspicious"
                decision_source = "model+trusted_softener"

    return verdict, prob, phishing_threshold, suspicious_threshold, decision_source, cert_info


def predict_url(url: str) -> Dict[str, Any]:
    try:
        normalized = normalize_url(url)
        train_cols = load_train_cols()
        bundle = load_bundle()

        verdict, prob, phishing_threshold, suspicious_threshold, decision_source, cert_info = predict_with_bundle(
            url=normalized,
            bundle=bundle,
            train_cols=train_cols,
        )
        signals = build_human_signals(
            url=normalized,
            verdict=verdict,
            prob=prob,
            cert_info=cert_info,
            decision_source=decision_source,
        )  
        ssl_status = None
        if cert_info:
            if cert_info.get("cert_valid") is False:
                ssl_status = "invalid"
            elif cert_info.get("cert_expired") is True:
                ssl_status = "expired"
            elif cert_info.get("cert_valid") is True:
                ssl_status = "valid"
            else:
                ssl_status = cert_info.get("error")

        ai_feedback = safe_generate_ai_feedback(
            url=normalized,
            classification=verdict,
            prediction_score=prob,
            top_signals=signals,
            trusted_domain_match=is_trusted_host(normalized),
            ssl_status=ssl_status,
            decision_source=decision_source,
        )

        # Optional logging
        try:
            log_prediction(normalized, prob, verdict, source="extension")
        except Exception:
            pass

        return {
            "success": True,
            "url": normalized,
            "classification": verdict,
            "prediction_score": round(prob, 6),
            "phishing_threshold": phishing_threshold,
            "suspicious_threshold": suspicious_threshold,
            "should_warn": verdict in {"suspicious", "phishing"},
            "should_block": verdict == "phishing",
            "decision_source": decision_source,
            "certificate_check": cert_info,
            "signals": signals,
            "ai_feedback": ai_feedback,
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
            "certificate_check": None,
            "signals": [],
            "ai_feedback": "The system could not analyze this URL.",
            "error": str(e),
        }


# =========================================================
# OPTIONAL QUICK TEST
# =========================================================
if __name__ == "__main__":
    test_urls = [
    # Clearly legitimate
    "https://www.google.com",
    "https://github.com",
    "https://docs.github.com/en",
    "https://www.microsoft.com/en-us/security",
    "https://support.apple.com",
    "https://www.cloudflare.com",
    "https://www.mozilla.org/en-US/firefox/new/",
    "https://www.wikipedia.org",
    "https://www.nytimes.com",
    "https://www.bbc.com/news",
    "https://www.cisa.gov",
    "https://www.ftc.gov",
    "https://www.irs.gov",
    "https://www.stanford.edu",
    "https://www.mit.edu",

    # Legitimate but auth/account-looking
    "https://accounts.google.com",
    "https://login.microsoftonline.com",
    "https://github.com/login",
    "https://appleid.apple.com",
    "https://www.paypal.com/signin",
    "https://www.amazon.com/ap/signin",
    "https://www.dropbox.com/login",
    "https://www.linkedin.com/login",
    "https://www.facebook.com/login",
    "https://www.instagram.com/accounts/login/",

    # Suspicious-looking synthetic phishing-style URLs
    "http://paypal-login-secure.ru/verify/account",
    "http://appleid-confirm-login.com/security/update",
    "http://microsoft-verify-account.net/login",
    "http://amazon-billing-update.info/confirm/payment",
    "http://netflix-account-locked.com/login/verify",
    "http://chase-secure-verification.com/account/login",
    "http://bankofamerica-alerts-login.net/verify",
    "http://coinbase-wallet-verify.org/secure",
    "http://docusign-review-document-login.com/verify",
    "http://office365-password-reset-support.com/login",

    # Typosquatting / brand impersonation
    "https://g00gle.com",
    "https://paypaI.com",  # capital i instead of lowercase L visually
    "https://micros0ft.com",
    "https://faceboook.com",
    "https://amaz0n-login.com",
    "https://netfIix.com",
    "https://github-security-alert.com",
    "https://apple-support-billing.com",
    "https://steamcommunnity.com",
    "https://disc0rd.com",

    # URL shorteners / redirects
    "https://bit.ly/3example",
    "https://tinyurl.com/login-update",
    "https://t.co/security-alert",
    "https://goo.gl/verify-account",
    "https://ow.ly/account-check",
    "https://is.gd/password-reset",
    "https://cutt.ly/paypal-verify",
    "https://rebrand.ly/secure-login",

    # IP-address hosts
    "http://192.168.1.1/login",
    "http://10.0.0.1/admin",
    "http://172.16.0.10/verify",
    "http://185.199.108.153/login",
    "http://8.8.8.8/account/update",
    "https://1.1.1.1",

    # Long / encoded / noisy URLs
    "http://example.com/login?session=abc123&redirect=http%3A%2F%2Fpaypal.com%2Fsignin",
    "http://secure-login.example.com/account/verify/update/password/reset",
    "http://example.com/%2F%2Fpaypal.com%2Fsignin",
    "http://example.com/login.php?email=user@example.com&token=1234567890abcdef",
    "http://account-update.example.net/verify?next=https%3A%2F%2Fbank.com",
    "http://xn--pple-43d.com/login",
    "http://xn--googl-fsa.com/security-check",

    # Hosted-content platforms, mixed risk
    "https://sites.google.com/view/account-security-check",
    "https://docs.google.com/forms/d/e/1FAIpQLSc-example/viewform",
    "https://github.io/login-verify",
    "https://raw.githubusercontent.com/user/repo/main/login.html",
    "https://storage.googleapis.com/example-bucket/index.html",
    "https://s3.amazonaws.com/example-bucket/login.html",
    "https://pastebin.com/raw/example",
    "https://medium.com/@user/security-awareness",

    # Discord / social invite cases
    "https://discord.gg/example",
    "https://discord.com/invite/example",
    "https://www.linkedin.com/in/example",
    "https://twitter.com/example",
    "https://x.com/example",
    "https://facebook.com/security",

    # Edge cases
    "google.com",
    "github.com/login",
    "paypal.com.signin.verify-account.example.com",
    "https://example.com",
    "http://localhost:3000",
    "not a url",
    "",
    ]


    for test_url in test_urls:
        result = predict_url(test_url)
        print(
                result["url"],
                "=>",
                result["classification"],
                "score=",
                result["prediction_score"],
                "source=",
                result["decision_source"],
            )
