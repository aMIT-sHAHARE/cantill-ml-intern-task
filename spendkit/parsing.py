"""Robust parsers for numbers, VAT rates and dates (messy ERP exports)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_NUM_CLEAN = re.compile(r"[^\d,.\-+()]")


def parse_number(x):
    """'5,266.63' '1.234,56' '(120)' '120-' 'R 1 234.5' -> float, else NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    s = _NUM_CLEAN.sub("", s.replace(" ", " "))
    if s.startswith("-"):
        neg, s = True, s[1:]
    s = s.replace("+", "").replace("(", "").replace(")", "").replace("-", "")
    if not s:
        return np.nan
    if "," in s and "." in s:
        # whichever comes last is the decimal separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        # '1,234' or '1,234,567' -> thousands ; '12,5' -> decimal comma
        if all(len(p) == 3 for p in parts[1:]) and len(parts) >= 2 and len(parts[0]) <= 3:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".") if len(parts) == 2 else s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_number_series(s: pd.Series) -> pd.Series:
    """Vectorised via unique values (fast on 100k+ rows)."""
    u = pd.Series(s.dropna().unique())
    m = dict(zip(u, u.map(parse_number)))
    return pd.to_numeric(s.map(m), errors="coerce").astype(float)


def parse_vat(x, default=np.nan):
    """'15' '0.15' '15%' '' '0' -> fraction (0.15). Blank -> default. '0' is a real 0."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    s = str(x).strip().replace("%", "")
    if not s:
        return default
    v = parse_number(s)
    if np.isnan(v):
        return default
    return v / 100.0 if v > 1 else v


def parse_vat_series(s: pd.Series, default: float) -> pd.Series:
    u = pd.Series(s.dropna().unique())
    m = dict(zip(u, u.map(lambda v: parse_vat(v, default))))
    return pd.to_numeric(s.map(m), errors="coerce").fillna(default).astype(float)


_DEFAULT_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%b-%y", "%d-%b-%Y", "%d/%m/%Y", "%Y/%m/%d",
    "%d.%m.%Y", "%Y%m%d", "%d-%m-%Y", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y",
]


def parse_dates(s: pd.Series, formats=None, dayfirst=True, valid_years=(1900, 2200)) -> pd.Series:
    """Try ISO + configured formats, then a dayfirst/monthfirst fallback.
    Anything outside valid_years becomes NaT (junk dates like 2018 / 2027)."""
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = s.notna() & (s != "")
    for fmt in list(formats or []) + _DEFAULT_FORMATS:
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            out.loc[parsed[ok].index] = parsed[ok].astype("datetime64[ns]")
            todo.loc[parsed[ok].index] = False
    if todo.any():
        try:
            parsed = pd.to_datetime(s[todo], errors="coerce", dayfirst=dayfirst, format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(s[todo], errors="coerce", dayfirst=dayfirst)
        ok = parsed.notna()
        out.loc[parsed[ok].index] = parsed[ok].astype("datetime64[ns]")
    lo, hi = valid_years
    bad = out.notna() & ((out.dt.year < lo) | (out.dt.year > hi))
    out[bad] = pd.NaT
    return out