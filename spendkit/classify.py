"""Classify every included line into the company's category tree (to the lowest level).

* The tree is read at runtime from the config's taxonomy file: NO category names in code.
* With labelled examples (optional): supervised model + exact material-code lookup.
* With ZERO labels: lexical match of item text to the leaf path text (TF-IDF, optionally
  multilingual embeddings) gives seeds -> self-training linear model on text + supplier + GL +
  material-code features -> (optional) LLM for the biggest low-confidence items.
* Unique "profiles" (text + supplier + code + GL ...) are classified once and propagated to rows.
* confidence in [0,1] = probability, discounted when the runner-up is close.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

from .config import load_config
from .dedup import norm_supplier
from .ingest import ingest, read_csv_safe, resolve

log = logging.getLogger("spendkit.classify")
MAXL = 4


# ----------------------------------------------------------------------------- taxonomy
def load_taxonomy(cfg: dict, base_dir=".") -> tuple[list[tuple], list[str]]:
    """-> (leaves, level_columns). A leaf is a tuple of 1..4 names; only true leaves are kept."""
    tc = cfg["taxonomy"]
    d = read_csv_safe(resolve(cfg.get("data_dir", base_dir), tc["file"]))
    cols = tc.get("level_columns")
    if not cols:
        lv = [c for c in d.columns if re.search(r"(^|_)(l|level)\s*_?\d", c.lower()) or "categor" in c.lower()]
        cols = lv or list(d.columns)
    cols = cols[:MAXL]
    paths = set()
    for row in d[cols].itertuples(index=False):
        p = []
        for v in row:
            v = None if pd.isna(v) else str(v).strip()
            if not v:
                break
            p.append(v)
        if p:
            paths.add(tuple(p))
    if not paths:
        raise ValueError("taxonomy file has no usable rows")
    # keep only true leaves (a path that is a strict prefix of another is an inner node)
    leaves = sorted(p for p in paths if not any(q != p and q[: len(p)] == p for q in paths))
    return leaves, cols


# ----------------------------------------------------------------------------- features
_SEP = re.compile(r"[;,:/|()_\-\.\[\]{}]+")


def clean_text(s) -> str:
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", _SEP.sub(" ", str(s).lower())).strip()


def _tok(prefix, v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    v = re.sub(r"[^a-z0-9]+", "", str(v).lower())
    return f"{prefix}_{v}" if v else ""


def featurize(df: pd.DataFrame, scfg: dict) -> pd.DataFrame:
    """-> frame with text, meta, code, pkey (profile key) for each row of df."""
    out = pd.DataFrame(index=df.index)
    out["text"] = df["description"].map(clean_text)
    sup = []
    for sid, nm in zip(df["supplier_id"], df["supplier_name"]):
        n = norm_supplier(nm, scfg["legal_suffixes"])[1] if pd.notna(nm) else ""
        sup.append(_tok("s", n) if n else _tok("s", sid))
    out["sup"] = sup
    out["code"] = df["item_code"].map(lambda v: _tok("c", v))
    out["gl"] = df["gl_desc"].map(lambda v: _tok("g", v)) if "gl_desc" in df else ""
    ctx = df["context"].fillna("").map(lambda v: " ".join(t for t in (_tok("x", w) for w in str(v).split(" | ")) if t))
    out["ctx"] = ctx
    out["doc"] = df["doc_type"].map(lambda v: _tok("d", v)) if "doc_type" in df else ""
    out["meta"] = (out["sup"] + " " + out["code"] + " " + out["gl"] + " " + out["ctx"] + " " + out["doc"]).str.strip()
    out["pkey"] = out["text"] + "\x1f" + out["meta"]
    return out


class Features:
    """Shared vectorisers (fit once on train+run+leaf text)."""

    def __init__(self, texts: list[str], metas: list[str]):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=150000)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=150000)
        self.m = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=1, sublinear_tf=True)
        self.w.fit(texts)
        self.c.fit(texts)
        self.m.fit(metas)

    def text_mat(self, texts):
        return normalize(sp.hstack([self.w.transform(texts), self.c.transform(texts)]).tocsr())

    def full(self, texts, metas):
        return sp.hstack([self.text_mat(texts), 0.7 * self.m.transform(metas)]).tocsr()


def _embed(texts, model_name):
    """Optional multilingual embeddings (CPU). Returns None if the package/model is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        m = SentenceTransformer(model_name)
        return m.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        log.warning("embeddings unavailable (%s); using TF-IDF only", type(e).__name__)
        return None


# ----------------------------------------------------------------------------- labelled data
def load_labelled(cfg: dict, base_dir, run_df: pd.DataFrame | None, leaves) -> pd.DataFrame | None:
    lc = cfg.get("labels")
    if not lc or not cfg["classify"].get("use_labels", True):
        return None
    try:
        if lc.get("train_from"):
            tcfg = load_config(lc["train_from"])
            frame = ingest(tcfg, tcfg.get("data_dir", base_dir))
            lab_dir = tcfg.get("data_dir", base_dir)
            lab_file = (tcfg.get("labels") or lc)["file"]
        else:
            frame, lab_dir, lab_file = run_df, cfg.get("data_dir", base_dir), lc["file"]
        lab = read_csv_safe(resolve(lab_dir, lab_file))
    except Exception as e:  # noqa: BLE001
        log.warning("labels not usable (%s); continuing zero-label", e)
        return None
    idc = lc.get("id_column", "record_id")
    cols = lc.get("level_columns") or [c for c in lab.columns if c.lower().startswith("category")]
    index = {p: i for i, p in enumerate(leaves)}

    def path(row):
        p = []
        for v in row:
            if pd.isna(v) or not str(v).strip():
                break
            p.append(str(v).strip())
        return index.get(tuple(p), -1)

    lab = lab.assign(leaf=[path(r) for r in lab[cols].itertuples(index=False)])
    lab = lab[lab.leaf >= 0][[idc, "leaf"]].drop_duplicates(idc)
    m = frame.merge(lab, left_on="record_id", right_on=idc, how="inner")
    if "include" in frame.columns:
        m = m[m["include"].fillna(1).astype(int) == 1] if "include" in m else m
    log.info("labelled examples usable: %d", len(m))
    return m if len(m) >= 50 else None


# ----------------------------------------------------------------------------- core
def _fit(X, y, w=None, seed=0):
    clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=40, tol=1e-4, random_state=seed)
    clf.fit(X, y, sample_weight=w)
    return clf


def _proba_full(clf, X, n_leaves):
    P = np.zeros((X.shape[0], n_leaves))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def _conf(P):
    top2 = np.sort(P, axis=1)[:, -2:]
    p1, p2 = top2[:, 1], top2[:, 0]
    pn = p1 / np.maximum(P.sum(1), 1e-9)  # OvR scores -> normalised
    return np.clip(0.6 * pn + 0.4 * np.clip(pn - p2 / np.maximum(P.sum(1), 1e-9), 0, 1), 0.02, 0.99)


def _llm_settings(cfg: dict) -> dict:
    """Config + environment overrides configured for Groq via LangChain."""
    lc = dict(cfg["classify"]["llm"])
    env = os.environ
    lc["provider"] = env.get("SPENDKIT_LLM_PROVIDER", "groq")
    # Available free models: "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
    lc["model"] = env.get("SPENDKIT_LLM_MODEL", "openai/gpt-oss-safeguard-20b")
    lc["api_key_env"] = env.get("SPENDKIT_LLM_KEY_ENV", "GROQ_API_KEY")
    return lc

def _make_chat(lc: dict):
    """-> callable(prompt) -> (text, tokens_in, tokens_out), or None when Groq is unavailable."""
    key = os.environ.get(lc["api_key_env"], "")
    if not key:
        log.info("GROQ_API_KEY not found in environment")
        return None
    try:
        from langchain_groq import ChatGroq  # noqa: PLC0415

        llm = ChatGroq(
            model=lc["model"],
            api_key=key,
            temperature=0,
            max_tokens=2048,
            timeout=int(lc.get("timeout", 120)),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("ChatGroq unavailable (%s)", type(e).__name__)
        return None

    def chat(prompt: str):
        response = llm.invoke(prompt)
        txt = response.content if isinstance(response.content, str) else str(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        tin = usage.get("input_tokens", 0)
        tout = usage.get("output_tokens", 0)
        return txt, tin, tout

    return chat


def _llm(profiles: pd.DataFrame, leaves, cfg, stats) -> dict[int, int]:
    """Label unique items with Groq LLM (optional). Never raises: failures just mean fewer seeds."""
    lc = _llm_settings(cfg)
    if not lc.get("enabled"):
        return {}
    chat = _make_chat(lc)
    if chat is None:
        log.info("no Groq LLM provider available -> local models only")
        return {}
    tree = "\n".join(f"{i}: {' > '.join(p)}" for i, p in enumerate(leaves))
    # Cap batch size to 8 so Groq free-tier payload/TPM limit (HTTP 413) is never exceeded
    out, bs = {}, min(int(lc.get("batch", 8)), 8)
    ids = list(profiles.index)
    fails = 0
    for s in range(0, len(ids), bs):
        chunk = ids[s : s + bs]
        items = "\n".join(
            f"{j}: {profiles.at[i, 'desc'][:120]} | sup={profiles.at[i, 'sup_name'][:35]} | {profiles.at[i, 'ctx'][:45]}"
            for j, i in enumerate(chunk)
        )
        prompt = (
            "Classify each purchase line into ONE leaf index from the category tree.\n"
            f"TREE:\n{tree}\n\nLINES:\n{items}\n\n"
            'Return ONLY valid JSON list: [{"n":0,"leaf":12},...]'
        )
        for attempt in range(3):
            try:
                txt, tin, tout = chat(prompt)
                stats["llm_in"] += tin or 0
                stats["llm_out"] += tout or 0
                m = re.search(r"\[.*\]", txt or "", re.S)
                if not m:
                    raise ValueError(f"No JSON array in LLM output: {txt[:120]!r}")
                for o in json.loads(m.group(0)):
                    j, lf = int(o["n"]), int(o["leaf"])
                    if 0 <= j < len(chunk) and 0 <= lf < len(leaves):
                        out[chunk[j]] = lf
                stats["llm_lines"] += len(chunk)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("Groq LLM batch failed (%s: %s) attempt %d", type(e).__name__, e, attempt + 1)
                time.sleep(2.5 * (attempt + 1))
        else:
            fails += 1
            if fails >= 5 and not out:
                log.warning("Groq LLM disabled for this run after repeated failures")
                break
        time.sleep(max(float(lc.get("delay", 0)), 1.5))
    return out

# def _make_chat(lc: dict):
#     """-> callable(prompt) -> (text, tokens_in, tokens_out), or None when Groq is unavailable."""
#     key = os.environ.get(lc["api_key_env"], "")
#     if not key:
#         log.info("GROQ_API_KEY not found in environment")
#         return None
#     try:
#         from langchain_groq import ChatGroq  # noqa: PLC0415

#         llm = ChatGroq(
#             model=lc["model"],
#             api_key=key,
#             temperature=0,
#             max_tokens=1500,
#             timeout=int(lc.get("timeout", 120)),
#         )
#     except Exception as e:  # noqa: BLE001
#         log.warning("ChatGroq unavailable (%s)", type(e).__name__)
#         return None

#     def chat(prompt: str):
#         response = llm.invoke(prompt)
#         txt = response.content if isinstance(response.content, str) else str(response.content)
#         usage = getattr(response, "usage_metadata", None) or {}
#         tin = usage.get("input_tokens", 0)
#         tout = usage.get("output_tokens", 0)
#         return txt, tin, tout

#     return chat


# def _llm(profiles: pd.DataFrame, leaves, cfg, stats) -> dict[int, int]:
#     """Label unique items with Groq LLM (optional). Never raises: failures just mean fewer seeds."""
#     lc = _llm_settings(cfg)
#     if not lc.get("enabled"):
#         return {}
#     chat = _make_chat(lc)
#     if chat is None:
#         log.info("no Groq LLM provider available -> local models only")
#         return {}
#     tree = "\n".join(f"{i}: {' > '.join(p)}" for i, p in enumerate(leaves))
#     out, bs = {}, int(lc["batch"])
#     ids = list(profiles.index)
#     fails = 0
#     for s in range(0, len(ids), bs):
#         chunk = ids[s : s + bs]
#         items = "\n".join(
#             f"{j}: {profiles.at[i, 'desc'][:160]} | supplier={profiles.at[i, 'sup_name'][:50]} | {profiles.at[i, 'ctx'][:60]}"
#             for j, i in enumerate(chunk)
#         )
#         prompt = (
#             "Classify each purchase line into exactly ONE leaf of this category tree.\n"
#             f"CATEGORIES (index: path):\n{tree}\n\nLINES (n: text):\n{items}\n\n"
#             'Answer ONLY with a JSON list like [{"n":0,"leaf":12},...] using the category index.'
#         )
#         for attempt in range(3):  # retry: free tiers rate-limit (HTTP 429)
#             try:
#                 txt, tin, tout = chat(prompt)
#                 stats["llm_in"] += tin or 0
#                 stats["llm_out"] += tout or 0
#                 for o in json.loads(re.search(r"\[.*\]", txt, re.S).group(0)):
#                     j, lf = int(o["n"]), int(o["leaf"])
#                     if 0 <= j < len(chunk) and 0 <= lf < len(leaves):
#                         out[chunk[j]] = lf
#                 stats["llm_lines"] += len(chunk)
#                 break
#             except Exception as e:  # noqa: BLE001 - never let the API break an unattended run
#                 log.warning("Groq LLM batch failed (%s) attempt %d", type(e).__name__, attempt + 1)
#                 time.sleep(2.0 * (attempt + 1))
#         else:
#             fails += 1
#             if fails >= 5 and not out:  # provider clearly unusable: stop wasting time
#                 log.warning("Groq LLM disabled for this run after repeated failures")
#                 break
#         if lc.get("delay"):
#             time.sleep(float(lc["delay"]))
#     return out



def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _group_arrays(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Candidate 'same category' groupings present in the data (PO, supplier, item, invoice)."""
    po = d["po_norm"] if "po_norm" in d else d["po"]
    out = {}
    for name, ser in (
        ("po", po),
        ("supplier", d["supplier_key"]),
        ("item", d["item_key"]),
        ("invoice", d["invoice_number"]),
    ):
        s = ser.astype("string").str.strip()
        s = s.mask(s.isin(["", "unknown-supplier", "unspecified-item"]))
        codes, _ = pd.factorize(s)
        if (codes >= 0).sum() > 50:
            out[name] = codes
    return out


def _smooth(P: np.ndarray, groups: dict, rng) -> tuple[np.ndarray, dict]:
    """Blend each row's class scores with the mean of its group(s). A grouping is trusted only as
    much as it is self-consistent: agreement of a row's top class with its group's leave-one-out
    top class, compared with a shuffled-group baseline (adaptive: no company-specific rule)."""
    acc, info = P.copy(), {}
    top, conf = P.argmax(1), P.max(1)
    strong = conf >= np.median(conf)
    for name, g in groups.items():
        ok = g >= 0
        G = g.max() + 1
        S = np.zeros((G, P.shape[1]))
        np.add.at(S, g[ok], P[ok])
        c = np.bincount(g[ok], minlength=G).astype(float)
        cnt = np.where(ok, c[np.where(ok, g, 0)], 0)
        multi = np.where(ok & (cnt >= 2))[0]
        if len(multi) < 50:
            continue
        loo = (S[g[multi]] - P[multi]) / (cnt[multi, None] - 1)
        sm = strong[multi]
        agree = float((loo.argmax(1) == top[multi])[sm].mean()) if sm.any() else 0.0
        gb = rng.permutation(g[multi])
        base = float(((S[gb] / c[gb, None]).argmax(1) == top[multi])[sm].mean()) if sm.any() else 0.0
        alpha = float(np.clip((agree - base) / (1 - base + 1e-9), 0, 1))
        info[name] = round(alpha, 3)
        if alpha > 0.05:
            acc[ok] += 3.0 * alpha * S[g[ok]] / c[g[ok], None]
    return acc / np.maximum(acc.sum(1, keepdims=True), 1e-9), info


def classify(df: pd.DataFrame, cfg: dict, base_dir=".", use_labels=True) -> tuple[pd.DataFrame, dict]:
    """Adds l1..l4 + confidence to df (only for include==1 rows). Returns (df, stats)."""
    t0 = time.time()
    df = df.copy()
    ccfg = cfg["classify"]
    leaves, _ = load_taxonomy(cfg, base_dir)
    L = len(leaves)
    for k in range(1, MAXL + 1):
        df[f"l{k}"] = ""
    df["confidence"] = np.nan
    stats = {"leaves": L, "mode": "zero_label", "llm_in": 0, "llm_out": 0, "llm_lines": 0}

    inc = df.index[df["include"].eq(1)]
    if len(inc) == 0:
        return df, stats
    F = featurize(df.loc[inc], cfg["suppliers"])
    spend = df.loc[inc, "spend_reporting"].abs().fillna(0.0)
    prof = (
        F.assign(spend=spend)
        .groupby("pkey", sort=False)
        .agg(
            text=("text", "first"),
            meta=("meta", "first"),
            code=("code", "first"),
            sup=("sup", "first"),
            n=("text", "size"),
            spend=("spend", "sum"),
        )
        .reset_index()
    )
    prof["desc"] = prof["text"]
    first = df.loc[inc].assign(_p=F["pkey"].to_numpy()).groupby("_p", sort=False).first()
    prof["sup_name"] = first["supplier_key"].reindex(prof["pkey"]).fillna("").astype(str).to_numpy()
    prof["ctx"] = (
        (
            first["context"].fillna("").astype(str)
            + " "
            + (first["gl_desc"].fillna("").astype(str) if "gl_desc" in first else "")
        )
        .reindex(prof["pkey"])
        .fillna("")
        .to_numpy()
    )
    leaf_txt = [" ".join(p).lower() for p in leaves]

    labelled = load_labelled(cfg, base_dir, df, leaves) if use_labels else None
    lab_prof = None
    if labelled is not None:
        LF = featurize(labelled, cfg["suppliers"])
        lab_prof = (
            LF.assign(leaf=labelled["leaf"].to_numpy())
            .groupby("pkey", sort=False)
            .agg(
                text=("text", "first"),
                meta=("meta", "first"),
                code=("code", "first"),
                leaf=("leaf", lambda s: s.value_counts().idxmax()),
                n=("leaf", "size"),
            )
            .reset_index()
        )

    feats = Features(
        list(prof.text) + (list(lab_prof.text) if lab_prof is not None else []) + leaf_txt,
        list(prof.meta) + (list(lab_prof.meta) if lab_prof is not None else []),
    )
    X = feats.full(prof.text, prof.meta)
    n = len(prof)

    if lab_prof is not None:
        # ---------------- supervised path
        stats["mode"] = "labelled"
        Xl = feats.full(lab_prof.text, lab_prof.meta)
        y = lab_prof.leaf.to_numpy()
        clf = _fit(Xl, y, w=np.sqrt(lab_prof.n.to_numpy()))
        P = _proba_full(clf, X, L)
        pred, conf = P.argmax(1), _conf(P)
        # exact material-code lookup (label purity >= 0.9)
        cl = (
            lab_prof[lab_prof.code != ""]
            .groupby("code")
            .apply(
                lambda g: pd.Series(
                    {
                        "leaf": g.groupby("leaf").n.sum().idxmax(),
                        "pur": g.groupby("leaf").n.sum().max() / g.n.sum(),
                    }
                ),
                include_groups=False,
            )
        )
        cl = cl[cl.pur >= 0.9]
        hit = prof.code.map(cl.leaf) if len(cl) else pd.Series(np.nan, index=prof.index)
        m = hit.notna().to_numpy()
        pred[m] = hit[m].astype(int).to_numpy()
        conf[m] = np.maximum(conf[m], 0.97)
        stats["code_lookup_profiles"] = int(m.sum())
    else:
        # ---------------- zero-label path: lexical seeds -> group smoothing -> self-training
        Xt = feats.text_mat(prof.text)
        Lm = feats.text_mat(leaf_txt)
        S = (Xt @ Lm.T).toarray()
        emb_name = ccfg.get("embedding_model")
        if ccfg.get("use_embeddings") and emb_name:
            E = _embed(list(prof.text) + leaf_txt, emb_name)
            if E is not None:
                S = 0.5 * S + 0.5 * np.clip(E[:n] @ E[n:].T, 0, 1)
                stats["embeddings"] = emb_name
        pkey_idx = {k: i for i, k in enumerate(prof.pkey)}
        ridx = np.array([pkey_idx[k] for k in F.pkey])  # row -> profile
        groups = _group_arrays(df.loc[inc])
        rng = np.random.default_rng(0)
        Pseed = _softmax(S[ridx] / 0.05)
        lcfg = ccfg["llm"]
        if lcfg.get("enabled"):
            top = prof.sort_values("spend", ascending=False).head(int(lcfg["max_items"]))
            got = _llm(top, leaves, cfg, stats)
            if got:
                arr_l = np.full(n, -1)
                for i, lf_ in got.items():
                    arr_l[i] = lf_
                m_ = arr_l[ridx] >= 0
                oh = np.full((m_.sum(), L), 0.1 / max(L - 1, 1))
                oh[np.arange(m_.sum()), arr_l[ridx][m_]] = 0.9
                Pseed[m_] = oh
                stats["llm_profiles"] = len(got)
        Pr, info = _smooth(Pseed, groups, rng)
        stats["group_alpha"] = info
        Xrow = X[ridx]
        Xleaf = feats.full(leaf_txt, [""] * L)
        for rnd in range(int(ccfg["self_train_rounds"]) + 1):
            conf_r = _conf(Pr)
            lab = Pr.argmax(1)
            q = 0.5 if rnd == 0 else 0.4
            take = []
            for k in range(L):  # per-leaf cap: no leaf swamps the seeds
                idx = np.where((lab == k) & (conf_r >= np.quantile(conf_r, q)))[0]
                take += list(idx[np.argsort(-conf_r[idx])][:1500])
            take = np.array(take, dtype=int)
            Xtr = sp.vstack([Xrow[take], Xleaf]).tocsr()
            ytr = np.concatenate([lab[take], np.arange(L)])
            w = np.concatenate([np.ones(len(take)), np.full(L, 5.0)])
            clf = _fit(Xtr, ytr, w)
            Pm = _proba_full(clf, Xrow, L)
            Pm = Pm / np.maximum(Pm.sum(1, keepdims=True), 1e-9)
            Pr, info = _smooth(0.6 * Pm + 0.4 * Pseed, groups, rng)
        row_lf, row_cf = Pr.argmax(1), _conf(Pr)
        # profile-level view (for the LLM step): spend-weighted majority of its rows
        pred = np.zeros(n, dtype=int)
        conf = np.zeros(n)
        tmp = pd.DataFrame({"p": ridx, "lf": row_lf, "cf": row_cf})
        for pi, g in tmp.groupby("p"):
            pred[pi] = g.lf.value_counts().idxmax()
            conf[pi] = g.cf.mean()

    # ---------------- optional LLM for the biggest low-confidence profiles
    lc = ccfg["llm"]
    if lc.get("enabled") and stats["mode"] == "labelled":
        cand = prof.index[(conf < float(lc["conf_below"]))]
        cand = prof.loc[cand].sort_values("spend", ascending=False).head(min(int(lc["max_items"]), 300))
        if len(cand):
            got = _llm(cand, leaves, cfg, stats)
            for i, lf in got.items():
                pred[i], conf[i] = lf, 0.85
            stats["llm_profiles"] = len(got)

    prof["leaf"], prof["conf"] = pred, conf

    # ---------------- map back to rows
    key2 = dict(zip(prof.pkey, zip(prof.leaf, prof.conf)))
    lf = np.array([key2[k][0] for k in F.pkey])
    cf = np.array([key2[k][1] for k in F.pkey], dtype=float)
    if stats["mode"] == "zero_label":
        lf, cf = row_lf, row_cf

    arr = np.array([list(p) + [""] * (MAXL - len(p)) for p in leaves], dtype=object)
    for k in range(MAXL):
        df.loc[inc, f"l{k + 1}"] = arr[lf, k]
    df.loc[inc, "confidence"] = np.round(cf, 3)
    stats["profiles"] = n
    stats["seconds"] = round(time.time() - t0, 1)
    return df, stats