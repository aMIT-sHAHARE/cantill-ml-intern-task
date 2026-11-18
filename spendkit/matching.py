"""Cross-system matching: the same purchase recorded in ERP and in the invoice tool.

Rule (brief): same purchase in both -> count the ERP record, mark the tool record DUPLICATE.
A tool invoice exported twice -> count once (extra copies DUPLICATE).
Matching is ONE-TO-ONE (ERP has identical repeated rows) and never depends on dates.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .dedup import norm_text


def _po_digits(v, rx):
    if v is None or pd.isna(v):
        return None
    m = rx.search(str(v))
    return m.group(0) if m else None


def _po_item(po_item, po_text):
    for v in (po_item,):
        if v is not None and pd.notna(v):
            try:
                return str(int(float(str(v))))
            except ValueError:
                pass
    if po_text is not None and pd.notna(po_text):
        m = re.search(r"/\s*(\d+)(?:\.\d+)?\s*$", str(po_text))
        if m:
            return str(int(m.group(1)))
    return None


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1.0)


def find_duplicates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    mc = cfg["matching"]
    df["matched_to"] = ""
    df["match_type"] = ""
    df["dup_conf"] = np.nan
    if not mc.get("enabled", True) or not (df.role.eq("primary").any() and df.role.eq("secondary").any()):
        return df
    tol = float(mc["amount_tolerance"])
    rx = re.compile(mc["po_regex"])
    default_vat = float(cfg["default_vat_rate"])

    sec = df.index[df.role.eq("secondary") & df.include.eq(1)]
    prim = df.index[df.role.eq("primary") & df.include.eq(1)]

    # ---- 1) exact re-exports inside the secondary system: keep the first, rest = DUPLICATE
    if mc.get("dedupe_secondary_exact", True):
        key = pd.DataFrame({
            "src": df.loc[sec, "source"], "inv": df.loc[sec, "invoice_number"], "sup": df.loc[sec, "supplier_key"],
            "po": df.loc[sec, "po"], "txt": df.loc[sec, "description"], "q": df.loc[sec, "quantity"],
            "a": df.loc[sec, "amount"], "d": df.loc[sec, "date"], "v": df.loc[sec, "vat_rate_given"]})
        dup = key.duplicated(keep="first")
        # only when an invoice number exists; otherwise identical lines may be genuine repeats
        dup &= key["inv"].notna()
        idx = dup[dup].index
        first = key.reset_index().drop_duplicates(list(key.columns), keep="first").set_index(
            list(key.columns))["index"]
        for i in idx:
            df.at[i, "include"] = 0
            df.at[i, "exclusion_reason"] = "DUPLICATE"
            df.at[i, "match_type"] = "tool_reexport"
            df.at[i, "matched_to"] = df.at[first.loc[tuple(key.loc[i])], "record_id"] if tuple(key.loc[i]) in first.index else ""
            df.at[i, "dup_conf"] = 0.95
        sec = df.index[df.role.eq("secondary") & df.include.eq(1)]

    # ---- 2) ERP <-> tool by PO (+ PO item) + net amount, one-to-one
    df["po_norm"] = [_po_digits(v, rx) for v in df["po"]]
    df["poi_norm"] = [_po_item(a, b) for a, b in zip(df["po_item"], df["po"])]
    erp_by_po = defaultdict(list)
    for i in prim:
        if df.at[i, "po_norm"]:
            erp_by_po[df.at[i, "po_norm"]].append(i)
    used = set()

    def amount_ok(t, e):
        """Tool net may be computed with default VAT; accept 0%/default if VAT was blank."""
        ea = df.at[e, "net_local"]
        cands = [df.at[t, "net_local"]]
        if pd.isna(df.at[t, "vat_rate_given"]):
            g = df.at[t, "amount"]
            cands += [g, g / (1 + default_vat)]
        d = min(_rel(c, ea) for c in cands if pd.notna(c)) if any(pd.notna(c) for c in cands) else 9
        return d

    def mark(t, e, mtype, conf):
        df.at[t, "include"] = 0
        df.at[t, "exclusion_reason"] = "DUPLICATE"
        df.at[t, "matched_to"] = df.at[e, "record_id"]
        df.at[t, "match_type"] = mtype
        df.at[t, "dup_conf"] = conf
        used.add(e)

    unmatched = []
    for t in sec:
        po = df.at[t, "po_norm"]
        best, bscore = None, None
        for e in erp_by_po.get(po, []) if po else []:
            if e in used:
                continue
            d = amount_ok(t, e)
            if d > tol:
                continue
            same_item = df.at[t, "poi_norm"] is not None and df.at[t, "poi_norm"] == df.at[e, "poi_norm"]
            qa, qb = df.at[t, "quantity"], df.at[e, "quantity"]
            same_q = pd.notna(qa) and pd.notna(qb) and abs(qa - qb) < 1e-6
            score = (0 if same_item else 1, 0 if same_q else 1, d)
            if bscore is None or score < bscore:
                best, bscore = e, score
        if best is not None:
            mark(t, best, "po_amount", 0.97 if bscore[0] == 0 else 0.93)
        else:
            unmatched.append(t)

    # ---- 3) fallback for rows with no/unknown PO: same supplier + net amount + quantity + text similarity
    min_sim = float(mc["min_text_sim_no_po"])
    pool = defaultdict(list)
    for i in prim:
        if i not in used:
            pool[df.at[i, "supplier_key"]].append(i)
    ntxt = df["description"].map(norm_text)
    for t in unmatched:
        cand = [e for e in pool.get(df.at[t, "supplier_key"], []) if e not in used]
        if not cand:
            continue
        ea = df.loc[cand, "net_local"].to_numpy(dtype=float)
        ta = df.at[t, "net_local"]
        if pd.isna(ta):
            continue
        close = np.abs(ea - ta) / np.maximum(np.abs(ea), 1.0) <= tol
        best, bs = None, 0.0
        for e in np.array(cand)[close]:
            qa, qb = df.at[t, "quantity"], df.at[e, "quantity"]
            if pd.notna(qa) and pd.notna(qb) and abs(qa - qb) > 1e-6:
                continue
            a, b = ntxt[t], ntxt[e]
            sim = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
            if sim > bs:
                best, bs = e, sim
        if best is not None and bs >= min_sim:
            mark(t, best, "fuzzy_supplier_amount_text", round(0.6 + 0.3 * bs, 3))
    return df