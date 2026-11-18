"""Decide what counts as spend: CANCELLED / NOT_APPROVED / NON_PROCUREMENT.

DUPLICATE is assigned later in matching.py.  All keywords are generic accounting
vocabulary (not company-specific) and can be extended per company in the config:

non_procurement:
  use_defaults: true
  patterns:                       # extra regexes
    - {field: gl_desc, regex: "^PAYROLL"}
    - {field: description, regex: "management fee"}
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# (field, regex).  Fields: gl_desc, description, supplier_name, doc_type
DEFAULT_PATTERNS = [
    # intercompany ledgers: 'IC06', 'IC Payables', 'I/C Payable', 'Intercompany ...'
    ("gl_desc", r"^\s*(ic|i/c)\s*[-_ ]?\d*\s*(payables?|receivables?|loan|charges?)?\s*$"),
    ("gl_desc", r"\binter[- ]?company\b|\bi/c\b|\bic (payables?|receivables?)\b"),
    ("description", r"\binter[- ]?company\b|\bi/c\b"),
    ("supplier_name", r"\binter[- ]?company\b"),
    # taxes (not goods/services)
    ("description", r"\b(sales tax|use tax|state (&|and) local tax(es)?|gst|vat|income tax|"
                    r"withholding tax|payroll tax|tax expense|tax payable|tax (input|output)|input tax|output tax)\b"),
    ("gl_desc", r"\b(vat|gst|sales tax|income tax|tax payable|withholding tax)\b"),
    # revenue / royalties / partner billings / financing
    ("gl_desc", r"\brevenue\b|\bsales income\b|\broyalt(y|ies)\b"),
    ("description", r"\broyalt(y|ies)\b|\brevenue\b"),
    ("description", r"\bpartner (billing|recharge)s?\b|\bjoint venture (billing|recharge)s?\b"),
    ("gl_desc", r"\bpartner (billing|recharge)s?\b"),
    ("description", r"\b(interest (expense|charges?|payable|income)|finance (cost|charge)s?|financing (cost|charge)s?|"
                    r"loan (repayment|interest)|dividends?)\b"),
    ("gl_desc", r"\b(interest (expense|charges?|payable|income)|finance (cost|charge)s?|financing|loan (repayment|interest)|dividends?)\b"),
]


def _compile(cfg: dict):
    np_cfg = cfg.get("non_procurement") or {}
    pats = list(DEFAULT_PATTERNS) if np_cfg.get("use_defaults", True) else []
    for p in np_cfg.get("patterns", []) or []:
        pats.append((p.get("field", "description"), p["regex"]))
    return [(f, re.compile(r, re.IGNORECASE)) for f, r in pats]


def apply_exclusions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["include"] = 1
    df["exclusion_reason"] = ""
    df["exclusion_conf"] = np.nan

    # 1) status -> CANCELLED / NOT_APPROVED.  Missing status = no status system -> keep.
    approved = {s.lower() for s in cfg["approved_statuses"]}
    cancel_kw = [k.lower() for k in cfg["cancelled_keywords"]]
    st = df["status_norm"]
    has = st.notna()
    ok = st.isin(approved) | ~has
    is_cancel = has & ~ok & st.fillna("").map(lambda v: any(k in v for k in cancel_kw))
    df.loc[~ok & ~is_cancel, "exclusion_reason"] = "NOT_APPROVED"
    df.loc[is_cancel, "exclusion_reason"] = "CANCELLED"
    df.loc[~ok, "exclusion_conf"] = 0.97

    # 2) NON_PROCUREMENT (only rows still included)
    todo = df["exclusion_reason"].eq("")
    hit = pd.Series(False, index=df.index)
    for field, rx in _compile(cfg):
        if field not in df.columns:
            continue
        col = df[field].astype("string").fillna("")
        hit |= todo & col.str.contains(rx, na=False)
    df.loc[hit, "exclusion_reason"] = "NON_PROCUREMENT"
    df.loc[hit, "exclusion_conf"] = 0.9

    # 3) un-priceable rows: kept as records but carry zero spend (flagged for review)
    df["include"] = df["exclusion_reason"].eq("").astype(int)
    return df