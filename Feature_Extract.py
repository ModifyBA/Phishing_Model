import json
import math
from urllib.parse import urlparse, parse_qs
import re
import ipaddress
from pathlib import Path
import tldextract



SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "account", "update",
    "password", "billing", "invoice", "secure"
}

TRUSTED_DOMAINS = {
    "google.com",
    "microsoft.com",
    "amazon.com",
    "facebook.com",
    "github.com"
}

SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co"
}

SUSPICIOUS_TLDS = {
    ".ru", ".tk", ".ml", ".ga", ".cf"
}

AUTH_FLOW_TERMS = {
    "oauth", "authorize", "token", "auth", "sso", "session",
    "redirect_uri", "response_type", "client_id", "scope",
    "state", "code_challenge", "code_verifier", "nonce",
    "openid", "oidc", "callback", "returnurl", "continue"
}
BASE_DIR = Path(__file__).resolve().parent
DOMAIN_COUNT_PATH = BASE_DIR / "domain_counts.json"

try:
    with open(DOMAIN_COUNT_PATH, "r", encoding="utf-8") as f:
        DOMAIN_COUNTS = json.load(f)
except Exception:
    DOMAIN_COUNTS = {}

#makes url normalization and parsing easier, also handles missing scheme cases
def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url

def get_hostname(parsed) -> str:
    return parsed.hostname.lower() if parsed.hostname else ""

def has_suspicious_subdomain(hostname: str) -> int:
    parts = hostname.split(".")
    if len(parts) < 3:
        return 0

    root = ".".join(parts[-2:])
    subdomains = parts[:-2]

    for sub in subdomains:
        for brand in ["paypal", "google", "amazon", "facebook", "chase"]:
            if brand in sub and brand not in root:
                return 1
    return 0

def get_domain_frequency(hostname: str) -> float:
    if not hostname:
        return 0.0

    ext = tldextract.extract(hostname)
    root_domain = ext.registered_domain  # e.g. google.com

    return math.log1p(DOMAIN_COUNTS.get(root_domain, 1))

def is_trusted_domain(hostname: str) -> int:
    return 1 if any(
        hostname == domain or hostname.endswith("." + domain) for domain in TRUSTED_DOMAINS
    ) else 0

def has_ip_address(hostname: str) -> int:
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0

#helper function to clean main extractor
def is_shortener(hostname: str) -> int:
    return 1 if hostname in SHORTENING_SERVICES else 0

def has_suspicious_tld(hostname: str) -> int:
    return 1 if any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0


def has_auth_flow_terms(parsed) -> int:
    blob = ((parsed.path or "") + "?" + (parsed.query or "")).lower()

    if any(term in blob for term in AUTH_FLOW_TERMS):
        return 1

    query_keys = parse_qs(parsed.query).keys()
    if any(key.lower() in AUTH_FLOW_TERMS for key in query_keys):
        return 1 
        
    return 0

# main extractor function
def extract_features(url:str) -> dict:
    url = normalize_url(url)
    parsed = urlparse(url)
    hostname = get_hostname(parsed)
    url_lower = url.lower()

    features = {}
    # -------------------------
    # Basic URL features
    # -------------------------
    features["valid_url"] = 1 if parsed.scheme in ["http", "https", "ftp"] and hostname else 0
    features["is_https"] = 1 if parsed.scheme == "https" else 0
    features["url_length"] = len(url)
    features["num_dots"] = hostname.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_at_symbols"] = url.count("@")
    features["num_slashes"] = url.count("/")
    features["num_question_marks"] = url.count("?")
    features["num_equals"] = url.count("=")
    features["num_ampersands"] = url.count("&")
    features["has_percent_encoding"] = 1 if "%" in url else 0

    # -------------------------
    # More complex features can be added here, such as:
    # -------------------------
    features["has_userinfo"] = 1 if (parsed.username is not None or parsed.password is not None) else 0
    features["special_char_count"] = sum(url.count(c) for c in ["@", "?", "&", "=", "-", "_", "#"])
    features["path_segment_count"] = len([seg for seg in parsed.path.split("/") if seg])
    features["percent_encoding_count"] = url.count("%")
    features["query_to_url_ratio"] = len(parsed.query) / max(1, len(url))
    features["path_to_url_ratio"] = len(parsed.path) / max(1, len(url))
    # -------------------------
    # Hostname/domain features
    # -------------------------
    features["hostname_length"] = len(hostname)
    parts = hostname.split(".") if hostname else []
    features["num_subdomains"] = max(0, len(parts) - 2)
    features["has_www"] = 1 if hostname.startswith("www.") else 0
    features["num_digits_in_host"] = sum(c.isdigit() for c in hostname)
    features["hostname_has_hyphen"] = 1 if "-" in hostname else 0
    features["has_ip_address"] = has_ip_address(hostname)
    features["is_trusted_domain"] = is_trusted_domain(hostname)
    features["is_shortener"] = is_shortener(hostname)
    features["suspicious_tld"] = has_suspicious_tld(hostname)
    features["has_punycode"] = 1 if "xn--" in hostname else 0
    features["domain_freq"] = get_domain_frequency(hostname)
    features["suspicious_subdomain"] = has_suspicious_subdomain(hostname)
    # -------------------------
    # Path/query features
    # -------------------------
    features["path_length"] = len(parsed.path)
    features["query_length"] = len(parsed.query)
    features["has_double_slash_in_path"] = 1 if "//" in parsed.path else 0
    features["num_query_params"] = len(parse_qs(parsed.query))
    features["has_port"] = 1 if parsed.port is not None else 0
    features["shortener_with_path"] = 1 if (
    features["is_shortener"] == 1 and features["path_length"] > 0
    ) else 0
    # -------------------------
    # Suspicious word features
    # -------------------------
    features["has_suspicious_words"] = 1 if any(
        word in url_lower for word in SUSPICIOUS_WORDS
    ) else 0

    features["suspicious_words_in_host"] = sum(
        hostname.count(word) for word in SUSPICIOUS_WORDS
    )

    path_query = ((parsed.path or "") + "?" + (parsed.query or "")).lower()
    features["suspicious_words_in_path_query"] = sum(
        path_query.count(word) for word in SUSPICIOUS_WORDS
    )

    BRAND_NAMES = {
    "google", "facebook", "amazon", "apple", "microsoft",
    "paypal", "netflix", "instagram"
}

    features["brand_in_url"] = 1 if any(b in url_lower for b in BRAND_NAMES) else 0
    features["brand_in_host"] = 1 if any(b in hostname for b in BRAND_NAMES) else 0
    # -------------------------
    # Auth / false-positive reduction
    # -------------------------
    features["auth_like_flow"] = has_auth_flow_terms(parsed)
    features["long_url"] = 1 if len(url) > 75 else 0

    return features

