"""Supplier and item de-duplication (entity resolution) with blocking, no O(n^2)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

try:  # optional speed-up
    from rapidfuzz import fuzz as _rf

    def _ratio(a: str, b: str) -> float:
        return _rf.ratio(a, b) / 100.0
except Exception:  # noqa: BLE001
    def _ratio(a: str, b: str) -> float:
        sm = SequenceMatcher(None, a, b)
        if sm.real_quick_ratio() < 0.8 or sm.quick_ratio() < 0.8:
            return 0.0
        return sm.ratio()

LEGAL = {"PTY", "LTD", "LIMITED", "CC", "INC", "LLC", "LLP", "GMBH", "BV", "AG", "SA", "CO", "CORP",
         "CORPORATION", "PLC", "THE", "NV", "SRL", "SPA", "SAS", "SARL", "OOO", "SDN", "BHD", "LP", "SE", "AB", "AS"}
GENERIC_TEXT = {"product", "products", "parts", "part", "item", "items", "misc", "miscellaneous", "service",
                "services", "n/a", "na", "none", "other", "general", "material", "materials", "various"}


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # deterministic root


def norm_supplier(name, extra_suffixes=()) -> tuple[str, str]:
    """-> (spaced, nospace) normalised name without legal suffixes / punctuation."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return "", ""
    s = re.sub(r"[^A-Z0-9 ]+", " ", str(name).upper().replace("&", " AND "))
    drop = LEGAL | {x.upper() for x in extra_suffixes}
    toks = [t for t in s.split() if t not in drop]
    if not toks:                        # name made only of suffix words: keep as is
        toks = s.split()
    return " ".join(toks), "".join(toks)


def _slug(s: str, n_tokens=6, maxlen=40) -> str:
    toks = re.sub(r"[^a-z0-9]+", " ", s.lower()).split()[:n_tokens]
    return "-".join(toks)[:maxlen].strip("-")


def _h(s: str, n=4) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def build_supplier_keys(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    scfg = cfg["suppliers"]
    thr = float(scfg["fuzzy_threshold"])
    sid = df["supplier_id"].astype("string").str.strip()
    nm = df["supplier_name"]
    pairs = pd.DataFrame({"sid": sid, "nm": nm}).drop_duplicates()
    pairs["spaced"], pairs["nos"] = zip(*pairs["nm"].map(lambda x: norm_supplier(x, scfg["legal_suffixes"]))) \
        if len(pairs) else ([], [])

    uf = UF()
    # id -> name clusters; ignore 'generic' ids that link too many different names
    names_per_id = pairs[pairs.sid.notna() & (pairs.nos != "")].groupby("sid").nos.nunique()
    generic_ids = set(names_per_id[names_per_id > 8].index)
    for r in pairs.itertuples():
        if r.nos:
            uf.find("n:" + r.nos)
        if pd.notna(r.sid):
            uf.find("i:" + r.sid)
            if r.nos and r.sid not in generic_ids:
                uf.union("n:" + r.nos, "i:" + r.sid)
    # fuzzy merge of names (blocked by 4-char prefix) catches typos: 'YARF IELD' vs 'YARFIELD'
    uniq = sorted({r.nos for r in pairs.itertuples() if len(r.nos) >= 6})
    blocks = defaultdict(list)
    for n in uniq:
        blocks[n[:4]].append(n)
    for b in blocks.values():
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                if _ratio(b[i], b[j]) >= thr:
                    uf.union("n:" + b[i], "n:" + b[j])

    # node per row
    def node(sid_, nos_):
        if nos_:
            return uf.find("n:" + nos_)
        if pd.notna(sid_):
            return uf.find("i:" + sid_)
        return None

    pairs["root"] = [node(a, b) for a, b in zip(pairs.sid, pairs.nos)]
    # human readable label per cluster: most frequent spaced name (else id)
    cnt = df.assign(sid=sid).groupby(["supplier_id", "supplier_name"], dropna=False).size().reset_index(name="n")
    cnt["spaced"] = cnt.supplier_name.map(lambda x: norm_supplier(x, scfg["legal_suffixes"])[0])
    cnt["nos"] = cnt.supplier_name.map(lambda x: norm_supplier(x, scfg["legal_suffixes"])[1])
    cnt["sid"] = cnt.supplier_id.astype("string").str.strip()
    cnt["root"] = [node(a, b) for a, b in zip(cnt.sid, cnt.nos)]
    label = {}
    for root, g in cnt.dropna(subset=["root"]).groupby("root"):
        g2 = g[g.spaced != ""]
        if len(g2):
            label[root] = g2.groupby("spaced").n.sum().idxmax().lower()
        else:
            label[root] = str(g.sid.dropna().iloc[0]).lower()
    # unique, deterministic labels
    seen, final = {}, {}
    for root in sorted(label):
        lab = label[root]
        if lab in seen:
            lab = f"{lab}-{_h(root)}"
        seen[lab] = root
        final[root] = lab
    key = pd.Series([final.get(node(a, b), "unknown-supplier") for a, b in
                     zip(sid, [norm_supplier(x, scfg["legal_suffixes"])[1] for x in nm])], index=df.index)
    # fast path above recomputes per row; cache via mapping to keep it O(n)
    df["supplier_key"] = key
    return df


def norm_text(s) -> str:
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(s).upper())).strip()


def build_item_keys(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Item = same material code OR same (informative) normalised description.
    Generic/blank texts ('Product', 'Parts', '') never merge items."""
    df = df.copy()
    icfg = cfg["items"]
    code = df["item_code"].astype("string").str.strip().str.upper()
    code = code.mask(code.isin(["", "NAN", "NONE", "NULL", "N/A", "0"]))
    txt = df["description"].map(norm_text)
    generic = {g.upper() for g in GENERIC_TEXT} | {g.upper() for g in icfg.get("generic_terms", [])}
    elig = (txt.str.len() >= int(icfg["min_text_len"])) & ~txt.isin(generic)
    # a text shared by too many codes is boilerplate, not an identity
    codes_per_text = pd.DataFrame({"t": txt[elig & code.notna()], "c": code[elig & code.notna()]}) \
        .groupby("t").c.nunique()
    boiler = set(codes_per_text[codes_per_text > int(icfg["max_codes_per_text"])].index)
    elig &= ~txt.isin(boiler)

    uf = UF()
    use_text = bool(icfg.get("text_merge", True))
    nodes = []
    for c, t, e in zip(code, txt, elig):
        has_c, has_t = pd.notna(c), bool(e)
        if has_c:
            uf.find("c:" + c)
        if has_t:
            uf.find("t:" + t)
        if has_c and has_t and use_text:
            uf.union("c:" + c, "t:" + t)
        nodes.append(("c:" + c) if has_c else (("t:" + t) if has_t else None))
    root = [uf.find(n) if n else None for n in nodes]
    r = pd.Series(root, index=df.index, dtype="object")
    # label cluster by its most frequent normalised text
    tmp = pd.DataFrame({"root": r, "txt": txt.where(txt != ""), "code": code})
    lab = {}
    for rt, g in tmp.dropna(subset=["root"]).groupby("root"):
        t = g.txt.dropna()
        lab[rt] = _slug(t.value_counts().idxmax()) if len(t) else _slug(str(g.code.dropna().iloc[0]))
    final, used = {}, {}
    for rt in sorted(lab):
        s = lab[rt] or "item"
        if s in used and used[s] != rt:
            s = f"{s}-{_h(rt)}"
        used[s] = rt
        final[rt] = s
    df["item_key"] = r.map(final).fillna("unspecified-item")
    return df