"""One command:  python -m spendkit.run --config configs/company_A.yaml --out submission/amit__A.csv

Pipeline: ingest -> net/FX spend -> supplier keys -> exclusions -> ERP/tool matching ->
item keys -> classification -> output file in the exact template format.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .amounts import add_spend
from .classify import classify
from .config import load_config
from .dedup import build_item_keys, build_supplier_keys
from .exclusions import apply_exclusions
from .ingest import ingest
from .matching import find_duplicates

log = logging.getLogger("spendkit")
OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    inc = df["include"].eq(1)
    out = pd.DataFrame({"record_id": df["record_id"]})
    out["include"] = df["include"].astype(int)
    out["exclusion_reason"] = np.where(inc, "", df["exclusion_reason"])
    out["supplier_key"] = df["supplier_key"]
    out["item_key"] = df["item_key"]
    sp = df["spend_reporting"].fillna(0.0).round(2)
    out["spend"] = np.where(inc, sp, 0.0)
    for k in range(1, 5):
        out[f"l{k}"] = np.where(inc, df[f"l{k}"].fillna(""), "")
    exc_conf = df["dup_conf"].fillna(df["exclusion_conf"]).fillna(0.95)
    out["confidence"] = np.where(inc, df["confidence"].fillna(0.3), exc_conf).round(3)
    return out[OUT_COLS]


def validate(df_in_n: int, out: pd.DataFrame, cfg: dict) -> list[str]:
    """Hard checks on the final file; returns a list of problems (empty = OK)."""
    bad = []
    if len(out) != df_in_n:
        bad.append(f"row count {len(out)} != input rows {df_in_n}")
    if out["record_id"].duplicated().any():
        bad.append("duplicate record_id values")
    if out.loc[out.include == 0, "spend"].ne(0).any():
        bad.append("excluded rows with non-zero spend")
    if out.loc[(out.include == 0) & (out.exclusion_reason == "")].shape[0]:
        bad.append("excluded rows without a reason")
    if out.loc[out.include == 1, "l1"].eq("").any():
        bad.append("included rows without l1")
    if out["confidence"].lt(0).any() or out["confidence"].gt(1).any():
        bad.append("confidence outside [0,1]")
    return bad


def run(cfg_path: str, out_path: str, data_dir: str | None = None, use_labels: bool = True,
        llm: bool = True, report: str | None = None) -> pd.DataFrame:
    t0 = time.time()
    cfg = load_config(cfg_path)
    if data_dir:
        cfg["data_dir"] = data_dir
    if not llm:
        cfg["classify"]["llm"]["enabled"] = False
    log.info("company=%s", cfg["name"])

    df = ingest(cfg, ".")
    n_in = len(df)
    log.info("ingested %d rows from %d sources", n_in, len(cfg["sources"]))
    df = add_spend(df, cfg, ".")
    df = build_supplier_keys(df, cfg)
    df = apply_exclusions(df, cfg)
    df = find_duplicates(df, cfg)
    df = build_item_keys(df, cfg)
    log.info("exclusions: %s", df.loc[df.include.eq(0), "exclusion_reason"].value_counts().to_dict())
    df, stats = classify(df, cfg, ".", use_labels=use_labels)
    log.info("classification: %s", stats)

    out = build_output(df)
    problems = validate(n_in, out, cfg)
    for p in problems:
        log.error("VALIDATION: %s", p)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")   # idempotent: always overwrites
    rep = {
        "rows": n_in, "included": int(out.include.sum()), "total_spend": float(out.spend.sum()),
        "exclusions": out.loc[out.include == 0, "exclusion_reason"].value_counts().to_dict(),
        "suppliers": int(out.supplier_key.nunique()), "items": int(out.item_key.nunique()),
        "fx_failed": int(df.fx_failed.sum()), "amount_failed": int(df.amount_failed.sum()),
        "date_failed": int(df.date_failed.sum()), "classify": stats, "problems": problems,
        "seconds": round(time.time() - t0, 1),
    }
    if report:
        Path(report).write_text(json.dumps(rep, indent=2, default=str))
    log.info("done in %.1fs -> %s | included %d/%d | spend %.2f", rep["seconds"], out_path,
             rep["included"], n_in, rep["total_spend"])
    if problems:
        raise SystemExit(f"{len(problems)} validation problem(s); see log")
    df.attrs["report"] = rep
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Config-driven spend cleaning + classification")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None, help="override config data_dir")
    ap.add_argument("--no-labels", action="store_true", help="ignore labelled examples (zero-label mode)")
    ap.add_argument("--no-llm", action="store_true", help="never call an LLM API")
    ap.add_argument("--report", default=None, help="optional JSON run report path")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    run(a.config, a.out, a.data_dir, not a.no_labels, not a.no_llm, a.report)


if __name__ == "__main__":
    main()
