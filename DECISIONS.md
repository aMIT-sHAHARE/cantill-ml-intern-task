# DECISIONS.md: findings, assumptions & trade-offs

How ambiguity was handled. Every number below was measured on the build data; nothing is estimated.
Legend: ✅ verified on data · 🧩 assumption (brief was silent or ambiguous) · ⚠️ known risk.

---

## 1 · Data findings

### Company A (mining, ZAR): 37,029 ERP lines + 28,520 invoice-tool lines

| # | Finding | Consequence in the tool |
|---|---|---|
| ✅ | ERP values are **net**; tool `gross_amount` is **gross** | `net = gross / (1 + VAT)` for tool lines (`amount_basis: gross`) |
| ✅ | VAT appears as `15`, `0.15`, `15%`, blank, `0` | One parser → fraction. Blank → default 15 % (brief). **`0` is a real zero-rating, not blank** |
| ✅ | Tool statuses: Paid 19,837 · Approved 8,180 · Cancelled 188 · On Hold 172 · Rejected 143 | Only approved/paid count. Cancelled → `CANCELLED`; On Hold / Rejected → `NOT_APPROVED` (315 lines) |
| ✅ | `po_reference` formats: `4510068836/10.0`, `PO#4510055507`, `PO 4510089105`, plain; ~18 % missing | PO extracted by config regex (`45\d{8}`), PO item parsed from `/10.0` |
| ✅ | ERP has **repeated identical rows** (same PO/value, different receipt doc) | Matching is strictly **one-to-one**; ERP rows are never duplicates of each other |
| ✅ | ERP `posting_date` blank on 66 % of rows, tool `invoice_date` on 65 % (43,030 undated rows overall) | Dates are only flagged; **matching never depends on dates** |
| ✅ | `document_currency` shows EUR / CHF / MZN on 16 ERP lines but `value_zar` is already ZAR | **Not converted** (would double-convert). `currency_fixed: ZAR` |
| ✅ | 434 ERP lines have non-positive value | Kept with their sign (credits / reversals) |

**Matching outcome:** of 28,520 tool lines, **22,677 are `DUPLICATE`** (16,651 PO + item + amount · 5,873 supplier + amount + quantity + text · 153 exact re-exports), 315 `NOT_APPROVED`, 188 `CANCELLED`, and **5,340 are tool-only spend** that stays included.

### Company B (metal packaging, USD, 2024): 41,028 lines, 8 ERPs (JDE / SAP), 25 currencies

| # | Finding | Consequence in the tool |
|---|---|---|
| ✅ | No supplier names, only `vendor_id` | Supplier key built from the id |
| ✅ | 3,343 non-USD lines have no `currency_rate` | Filled from `fx_rates_usd.csv` by invoice month (nearest month as fallback) |
| ✅ | `currency_rate` = units of line currency per 1 USD | `USD = amount / rate`. USD lines are already USD |
| ✅ | 833 negative amounts occur as **+/− pairs on the same invoice** (reversals) | Kept so pairs net to zero; never blindly dropped |
| ✅ | `invoice_date` contains junk years (2018, 2027 …) | `date: [invoice_date, posting_date]` with `valid_years: [2023, 2025]`, first *valid* date wins |
| ✅ | 16.7 % of descriptions blank; 3 lines have blank amount | Classified from supplier / GL / material-code features; blank amounts get spend 0 and are flagged |
| ✅ | NON_PROCUREMENT seeds are intercompany GL descriptions (`IC06`, `IC Payables`, `I/C Payable`, `IC05` …) and tax lines (`Sales Tax`, `State & Local Taxes`, `GST Input Tax`) | Generic accounting patterns (see §3). **450 / 450 seeds found, 0 false positives**; 2,491 of 41,028 lines excluded overall |

### Label artifacts in `labels_A.csv` (deliberately **not** replicated)

- **Supplier:** key `SG-f85711` holds **20,891 lines, 20,718 of them with a blank `vendor_id`, covering 120 different supplier names**. It looks like blank ids were grouped together by the labelling script. Outside this group my supplier grouping scores pairwise precision / recall **1.0 / 1.0**; with it, recall is 0.10 purely because of the artifact.
- **Item:** key `IG-06a8b34` merges **192 material codes** (mostly the free-text catch-all material). Outside it my item grouping scores precision / recall **0.94 / 0.995**.
- Decision: merging 120 unrelated suppliers contradicts the brief's own definition of de-duplication, so the tool follows the definition, not the artifact. ⚠️ If the hidden truth contains the same artifact, supplier-recall on A2 will look low for the same reason.
- Labels also show that **category is decided at PO level** (PO → L1 purity 100 %, supplier → 98 %), which motivated the group-smoothing step in §5.

---

## 2 · Design decisions

| Decision | Why | Alternative rejected |
|---|---|---|
| **Everything company-specific lives in YAML** | Brief: "new company = new config, not new code" | Per-company branches in code |
| **No category names / GL codes / vendor ids in code** | Must work on an unseen tree and industry | Hard-coding A/B vocabulary (would score 0 on C) |
| **Never drop an input record** | Output must be one row per input row, excluded ones too | Filtering excluded rows out |
| **Unique "profiles" classified once, propagated to rows** | Speed (<1 min / 25k lines) and consistency | Row-by-row classification |
| **Blocked fuzzy matching, union-find for entities** | Avoid O(n²), keep transitivity (`YARF IELD` ≡ `YARFIELD`) | All-pairs similarity |
| **Optional LLM, never required; failures never abort a run** | Brief requires a local/free fallback; unattended runs | Hard dependency on one paid API |
| **Fixed random seeds, atomic overwrite of the output** | Idempotent, byte-identical re-runs (verified A and B) | Non-deterministic sampling |
| **Output validated before exit** (row count, unique ids, spend 0 when excluded, reasons present, confidence in [0, 1]) | Catch silent format errors before submission | Trusting the pipeline |

---

## 3 · Exclusions: ordering & rules

1. **Status first:** not approved/paid → `CANCELLED` (cancel / void / delete) or `NOT_APPROVED`. Sources with no status column count as approved.
2. **NON_PROCUREMENT:** generic accounting vocabulary only, on GL description / text / supplier name: intercompany (`IC06`, `I/C`, `intercompany`), taxes (sales / use / GST / VAT / withholding …), revenue, royalties, partner billings / recharges, financing costs (interest expense, finance charges, loan repayment, dividends). Extra regexes per company go in `non_procurement.patterns`.
3. **DUPLICATE last** (needs supplier keys and amounts), applied only to lines still included.
4. 🧩 Rule order means a cancelled tool line keeps `CANCELLED` even if it also matches an ERP line.

---

## 4 · Matching assumptions

- 🧩 A tool line and an ERP line are **the same purchase** if PO (+ PO item) agree **and** net amounts agree within **1 %**. When the tool VAT is blank, a 0 % or 15 % reading is also accepted (the true rate is unknown).
- 🧩 If no PO matches, fall back to **same supplier + net amount (1 %) + equal quantity + text similarity ≥ 0.5**. Chosen to favour precision; a wrong match would silently remove real spend.
- 🧩 A tool invoice "exported twice" = **identical in every field except `tool_record_id`, and carrying an invoice number**. Identical lines without an invoice number may be genuine repeats and are kept.
- 🧩 Tool lines are matched **line-to-line**. ⚠️ A tool line that covers several ERP receipts (or part of one) is not split or summed and stays as tool-only spend.
- 🧩 Where several ERP candidates fit, prefer same PO item, then same quantity, then the smallest amount difference.
- ⚠️ **Company B:** duplicate matching is switched off (single system). 8,383 B lines share identical invoice / PO / item / amount / quantity / vendor fields; I did not verify them against labels (the seeds contain no `DUPLICATE` examples) and left them included rather than risk removing genuine multi-line invoices.

---

## 4b · Supplier & item keys

- **Supplier:** names normalised (upper-case, punctuation and legal suffixes such as `PTY / LTD / CC / GMBH / SA` removed) → union with `vendor_id` → blocked fuzzy merge (ratio ≥ 0.88, 4-character block). A vendor id linked to **more than 8** different names is treated as generic and not used to merge.
- **Item:** same material code **or** same informative description (≥ 15 characters, not generic like `Product` / `Parts`, shared by ≤ 150 codes). Thresholds were tuned on A's labels: pairwise F1 rose from 0.951 → **0.967** (outside the catch-all item).
- 🧩 Keys are readable slugs (`acme engineering`, `valve-2in-ball`) with a short hash suffix only on collisions. Lines with no code and no usable text get `unspecified-item`.

---

## 5 · Classification decisions

- **Leaves only:** a category that is a strict prefix of another path is an inner node and is never assigned. Shallower leaves leave deeper `l` columns blank. Names are copied from the tree file verbatim.
- **Labelled mode** (A / B / A2 / B2): linear model + exact material-code lookup (label purity ≥ 0.9). `labels.train_from` lets fresh data learn from the labelled build data.
- **Zero-label mode** (unseen company C): lexical + optional multilingual-embedding seeds → optional LLM labels for the top-spend unique items → **adaptive group smoothing** → 3 self-training rounds. Each grouping (PO / supplier / item / invoice) is trusted only as much as it is self-consistent compared with a shuffled-group baseline, so nothing company-specific is assumed.
- 🧩 **Confidence** = probability, discounted by the runner-up margin; **excluded rows** carry the confidence of the exclusion decision (0.97 status, 0.90 pattern, 0.6–0.97 match), following the template's example.
- 🧩 Excluded rows have blank `l1–l4` (as in the template).

---

## 6 · Known risks (stated plainly)

| Risk | Detail |
|---|---|
| ⚠️ **Zero-label accuracy** | Offline TF-IDF only: A L1 42.8 % / L3 14.0 %, B L1 53.1 % / L3 31.7 %. A's mining shorthand barely overlaps the tree names (seed accuracy 36.5 % vs 35.6 % majority). Embeddings and/or an LLM are expected to help, but the LLM path was verified with a **mock** only and no accuracy claim is made for it |
| ⚠️ **LLM free tiers** | May rate-limit (retries + `delay` setting provided); a failed call only means fewer seeds |
| ⚠️ **Fuzzy-match precision on A** | 5,873 tool lines were matched by supplier + amount + text with no PO; there is no ground truth for tool lines, so this precision is unmeasured |
| ⚠️ **Time-sensitive rules** | None; no date logic is used in matching or classification |

---

## 7 · Out of scope / future work

Splitting or summing partial invoices · cross-company supplier master · learning exclusion patterns from labelled seeds · hierarchical (level-by-level) classification with per-level confidence · active-learning loop for low-confidence, high-spend items.
