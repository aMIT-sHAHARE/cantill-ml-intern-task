"""Config loading and validation.

A company = one YAML/JSON file. No company-specific logic lives in code.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

# Canonical column names every source is mapped onto.
CANONICAL = [
    "record_id", "date", "supplier_id", "supplier_name", "description", "context",
    "amount", "quantity", "po", "po_item", "item_code", "invoice_number",
    "status", "vat_rate", "currency", "currency_rate", "gl_account", "gl_desc",
    "doc_type",
]
# These may be given as a list of columns (joined with ' | ', or first-valid for date).
LIST_OK = {"description", "context", "date"}


class ConfigError(ValueError):
    pass


DEFAULTS = {
    "reporting_currency": "USD",
    "default_vat_rate": 0.15,
    "valid_years": [2000, 2100],
    "fx": {"file": None},
    "approved_statuses": ["approved", "paid"],
    "cancelled_keywords": ["cancel", "void", "delet"],
    "non_procurement": {"use_defaults": True, "patterns": []},
    "matching": {
        "enabled": True,
        "po_regex": r"\d{8,12}",
        "amount_tolerance": 0.01,
        "min_text_sim_no_po": 0.5,
        "dedupe_secondary_exact": True,
    },
    "suppliers": {"legal_suffixes": [], "fuzzy_threshold": 0.88},
    "items": {"text_merge": True, "min_text_len": 15, "max_codes_per_text": 150, "generic_terms": []},
    "classify": {
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "use_embeddings": True,
        "use_labels": True,
        "self_train_rounds": 3,
        "item_vote": True,
        # "llm": {"enabled": True, "provider": "anthropic", "model": "claude-haiku-5-5",
        #         "base_url": None, "api_key_env": "ANTHROPIC_API_KEY", "delay": 0,
        #         "max_items": 1500, "batch": 20, "conf_below": 0.6},
        "llm": {"enabled": True, "provider": "groq", "model": "openai/gpt-oss-safeguard-20b",
                "base_url": None, "api_key_env": "GROQ_API_KEY", "delay": 1.5,
                "max_items": 1500, "batch": 8, "conf_below": 0.6},
    },
}

SOURCE_DEFAULTS = {
    "role": "primary",          # primary | secondary (secondary rows may duplicate primary rows)
    "id_prefix": "",
    "amount_basis": "net",      # net | gross
    "currency_fixed": None,     # amounts already in this currency (ignores any currency column)
    "date_formats": [],
    "date_dayfirst": True,
    "rate_convention": "units_per_usd",
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Cannot parse {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a mapping at top level")

    cfg = _merge(DEFAULTS, raw)
    cfg["_path"] = str(path)
    cfg.setdefault("name", path.stem)
    if not cfg.get("taxonomy", {}).get("file"):
        raise ConfigError("config.taxonomy.file is required")
    srcs = cfg.get("sources")
    if not srcs or not isinstance(srcs, list):
        raise ConfigError("config.sources must be a non-empty list")
    names = set()
    for i, s in enumerate(srcs):
        s2 = _merge(SOURCE_DEFAULTS, s)
        if "file" not in s2 or "columns" not in s2:
            raise ConfigError(f"sources[{i}] needs 'file' and 'columns'")
        s2.setdefault("name", f"src{i}")
        if s2["name"] in names:
            raise ConfigError(f"duplicate source name {s2['name']}")
        names.add(s2["name"])
        bad = [c for c in s2["columns"] if c not in CANONICAL]
        if bad:
            raise ConfigError(f"sources[{i}].columns has unknown canonical names {bad}; allowed: {CANONICAL}")
        if "record_id" not in s2["columns"] or "amount" not in s2["columns"]:
            raise ConfigError(f"sources[{i}].columns must map record_id and amount")
        if s2["role"] not in ("primary", "secondary"):
            raise ConfigError("source role must be primary or secondary")
        if s2["amount_basis"] not in ("net", "gross"):
            raise ConfigError("amount_basis must be net or gross")
        srcs[i] = s2
    cfg["reporting_currency"] = str(cfg["reporting_currency"]).upper()
    return cfg