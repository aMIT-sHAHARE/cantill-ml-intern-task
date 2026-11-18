"""Net-of-tax amounts and conversion to the reporting currency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import read_csv_safe, resolve
from .parsing import parse_number


def load_fx(cfg, base_dir) -> pd.DataFrame | None:
    f = (cfg.get("fx") or {}).get("file")
    if not f:
        return None
    d = read_csv_safe(resolve(cfg.get("data_dir", base_dir), f))
    low = {c.lower(): c for c in d.columns}
    cc, mc = low.get("currency"), low.get("month")
    rc = low.get("units_per_usd") or low.get("rate")
    if not (cc and mc and rc):
        raise ValueError(f"fx file needs currency, month, units_per_usd columns; has {list(d.columns)}")
    out = pd.DataFrame({
        "currency": d[cc].str.upper().str.strip(),
        "month": pd.PeriodIndex(pd.to_datetime(d[mc].astype(str).str[:7], format="%Y-%m", errors="coerce"), freq="M"),
        "rate": d[rc].map(parse_number),
    }).dropna()
    return out[out.rate > 0].drop_duplicates(["currency", "month"])


def add_spend(df: pd.DataFrame, cfg: dict, base_dir=".") -> pd.DataFrame:
    df = df.copy()
    rep = cfg["reporting_currency"]

    # 1) net of tax. gross sources: net = gross / (1+VAT), blank VAT -> default_vat_rate.
    gross = df["amount_basis"].eq("gross")
    df["net_local"] = np.where(gross, df["amount"] / (1.0 + df["vat_rate_used"]), df["amount"])

    # 2) currency -> reporting currency
    ccy = df["currency"].fillna(rep)
    df["currency_norm"] = ccy
    rate = pd.Series(np.nan, index=df.index)
    src = pd.Series("", index=df.index, dtype="object")

    same = ccy.eq(rep)
    rate[same], src[same] = 1.0, "reporting_ccy"

    # line-level rate: units of line currency per 1 USD (only valid when reporting is USD)
    if rep == "USD":
        lr = df["currency_rate"]
        ok = ~same & lr.notna() & (lr > 0)
        rate[ok], src[ok] = lr[ok], "line_rate"

    need = rate.isna()
    fx = load_fx(cfg, base_dir)
    if need.any() and fx is not None:
        tbl = {(c, m): r for c, m, r in zip(fx.currency, fx.month, fx.rate)}
        by_ccy = {c: g.sort_values("month") for c, g in fx.groupby("currency")}
        months = df["date"].dt.to_period("M")
        rep_per_usd = 1.0 if rep == "USD" else None  # cross rate via USD for non-USD reporting

        def lookup(c, m):
            if (c, m) in tbl:
                return tbl[(c, m)], "fx_table_month"
            g = by_ccy.get(c)
            if g is None:
                return np.nan, ""
            if pd.isna(m):
                return float(g.rate.median()), "fx_table_nearest"
            i = (g.month.astype("int64") - m.ordinal).abs().idxmin()
            return float(g.loc[i, "rate"]), "fx_table_nearest"

        for i in df.index[need]:
            c, m = ccy[i], months[i]
            r, s = lookup(c, m)
            if rep != "USD" and not np.isnan(r):
                rr, _ = lookup(rep, m)
                r = r / rr if rr and not np.isnan(rr) else np.nan
            if not np.isnan(r):
                rate[i], src[i] = r, s

    df["fx_rate_used"] = rate
    df["fx_source"] = src
    df["fx_failed"] = rate.isna()
    # USD = amount / rate (rate = units of currency per 1 reporting unit)
    df["spend_reporting"] = (df["net_local"] / rate).astype(float)
    return df