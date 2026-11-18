"""Load every configured source into ONE canonical DataFrame."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANONICAL, LIST_OK
from .parsing import parse_dates, parse_number_series, parse_vat_series


def resolve(base_dir: str | Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(base_dir) / q


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read as strings. Tries utf-8(-sig), then latin-1. Fails loudly if file missing."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    last = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False,
                               na_values=[""], on_bad_lines="warn", low_memory=False)
        except UnicodeDecodeError as e:
            last = e
    raise last  # pragma: no cover


def _clean_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s == "")


def _col(df: pd.DataFrame, name: str, src: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(f"Column '{name}' (from config source '{src}') not in file. Columns: {list(df.columns)[:40]}")
    return df[name]


def load_source(src: dict, cfg: dict, base_dir) -> pd.DataFrame:
    raw = read_csv_safe(resolve(base_dir, src["file"]))
    cols = src["columns"]
    consts = src.get("constants", {}) or {}
    out = pd.DataFrame(index=raw.index)
    vy = cfg["valid_years"]

    for c in CANONICAL:
        spec = cols.get(c)
        if c == "date":
            specs = [spec] if isinstance(spec, str) else (spec or [])
            d = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
            for name in specs:   # first VALID parsed date wins
                p = parse_dates(_col(raw, name, src["name"]), src["date_formats"], src["date_dayfirst"], vy)
                d = d.fillna(p)
            out["date"] = d
            out["date_failed"] = d.isna()
        elif spec is None:
            out[c] = pd.Series(consts.get(c), index=raw.index, dtype="string")
        elif isinstance(spec, list):
            if c not in LIST_OK:
                raise ValueError(f"{c} cannot be a list")
            parts = [_clean_str(_col(raw, n, src["name"])) for n in spec]
            j = pd.concat(parts, axis=1).apply(lambda r: " | ".join(x for x in r if pd.notna(x)), axis=1)
            out[c] = j.mask(j == "")
        else:
            out[c] = _clean_str(_col(raw, spec, src["name"]))

    if "date" not in cols:
        out["date"] = pd.NaT
        out["date_failed"] = True

    out["record_id"] = src["id_prefix"] + out["record_id"].astype("string")
    out["amount_raw"] = out["amount"]
    out["amount"] = parse_number_series(out["amount"])
    out["amount_failed"] = out["amount"].isna()
    out["quantity"] = parse_number_series(out["quantity"].astype("string")) if out["quantity"].notna().any() else np.nan
    out["currency_rate"] = parse_number_series(out["currency_rate"]) if out["currency_rate"].notna().any() else np.nan
    out["vat_rate_given"] = parse_vat_series(out["vat_rate"], np.nan) if out["vat_rate"].notna().any() else np.nan
    out["vat_rate_used"] = out["vat_rate_given"].fillna(cfg["default_vat_rate"])
    out["status_norm"] = out["status"].astype("string").str.lower().str.strip()
    out["source"] = src["name"]
    out["role"] = src["role"]
    out["amount_basis"] = src["amount_basis"]
    out["currency_fixed"] = src["currency_fixed"]
    if src["currency_fixed"]:
        out["currency"] = src["currency_fixed"]
    out["currency"] = out["currency"].astype("string").str.upper().str.strip()
    # keep a few extra raw columns (if configured) as classifier context
    for extra in src.get("keep_raw", []) or []:
        out["raw_" + extra] = _clean_str(_col(raw, extra, src["name"]))
    return out.reset_index(drop=True)


def ingest(cfg: dict, base_dir=".") -> pd.DataFrame:
    frames = [load_source(s, cfg, cfg.get("data_dir", base_dir)) for s in cfg["sources"]]
    df = pd.concat(frames, ignore_index=True)
    if df["record_id"].isna().any():
        raise ValueError("Some rows have a blank record_id; every input row needs an id")
    dup = df["record_id"].duplicated()
    if dup.any():
        # never drop an input record: make ids unique but keep a warning column
        df["record_id_dup"] = dup
        df.loc[dup, "record_id"] = df.loc[dup, "record_id"] + "#dup" + df[dup].groupby("record_id").cumcount().astype(str)
    df["row_no"] = np.arange(len(df))
    return df