# APPROACH

> `spendkit`: one config-driven pipeline for every company. New company = new YAML, never new code.

## AI usage (disclosure)
- **Build time:** Claude (Anthropic) was used as a coding and analysis assistant; I reviewed and tested all code.
- **Run time:** the tool runs **fully without any LLM** (TF-IDF, optional local `paraphrase-multilingual-MiniLM-L12-v2` embeddings). An LLM is an *optional* booster for zero-label classification: Anthropic, or any OpenAI-compatible endpoint (local Ollama, Groq / Gemini free tiers). Selected by environment variables; failures never abort a run.

## Pipeline
`ingest → net spend + FX → supplier keys → exclusions → ERP/tool matching → item keys → classify → validated output`

| Step | Design |
|---|---|
| **Clean & combine** | Config maps every source to one canonical schema. Robust parsing of numbers (`5,266.63`, `1.234,56`, `(120)`), VAT (`15`, `0.15`, `15%`, blank, real `0`) and dates (junk years rejected). |
| **Spend** | Net of tax: gross → `gross/(1+VAT)`, blank VAT → default. Currency → reporting currency via the line's own rate, else the monthly FX table. Reversal pairs are kept so they net to zero. |
| **Exclusions** | Status → `CANCELLED` / `NOT_APPROVED`. `NON_PROCUREMENT` via generic accounting patterns (intercompany, taxes, revenue, royalties, partner billings, financing) on GL description / text / supplier, extendable in config. Found 450 / 450 seeded cases in B, 0 false positives. |
| **Matching (A)** | Tool ↔ ERP is **one-to-one**: (1) PO + PO item + net amount ±1 % (VAT-blank tolerant); (2) fallback without PO: same supplier + net amount + quantity + text similarity. ERP counted, tool line `DUPLICATE`. Exact re-exports counted once. Dates are never used (66 % blank). |
| **Supplier dedup** | Normalised names (legal suffixes removed) + vendor ids joined with union-find, plus blocked fuzzy matching (no O(n²)). |
| **Item dedup** | Same material code **or** same informative description; generic / blank texts (`Product`, `Parts`) never merge. |

## Classifying into an unseen tree
The tree is read at runtime (up to 4 levels, true leaves only); no category names exist in code. Unique *profiles* (text + supplier + code + GL) are classified once and propagated.

1. **Seeds:** similarity between item text and each leaf path (TF-IDF word + char n-grams, optional multilingual embeddings).
2. **Optional LLM:** labels the top-spend unique items (default 1,500) from the tree; the local model propagates to the rest.
3. **Adaptive group smoothing:** lines sharing a PO / supplier / item / invoice are pulled toward a common class. Each grouping is trusted only as far as it is self-consistent versus a shuffled baseline, so no company-specific rule is assumed.
4. **Self-training:** 3 rounds of a linear model on text + supplier + GL + material-code features, with every leaf's own name as an anchor example.
5. **Confidence** = class probability, discounted when the runner-up is close. With labelled examples (A, B, A2, B2) a supervised model plus an exact material-code lookup is used instead.

## Accuracy: Company A with labels hidden
Zero-label run on all 37,029 labelled ERP lines (`--no-labels --no-llm`, offline TF-IDF only):

| Level | Accuracy | Spend-weighted |
|---|---|---|
| L1 (5 classes) | **42.8 %** | 59.2 % |
| L2 (24 classes) | **17.5 %** | 30.6 % |
| L3 (40 classes) | **14.0 %** | 24.3 % |

Majority-class baseline at L1 is 35.6 %. The lexical seed alone scores 36.5 %: A's mining shorthand (`ROD DRIL;SIZE 35 MM`) shares almost no words with the tree names, so the offline result is a **floor**, not the expected result with embeddings or an LLM.

Other checks: B with labels, 70/30 hold-out → L1 97.4 %, L3 96.6 % · B zero-label offline → L1 53.1 %, L3 31.7 % · A supplier grouping P/R 1.0/1.0, item grouping 0.94/0.995 (outside label artifacts, see DECISIONS.md). The LLM path was tested with a mock model only; no accuracy claim is made for it.

## Runtime
| Dataset | Lines | Time (no LLM) |
|---|---|---|
| Company A (ERP + tool) | 65,549 | **29 s** |
| Company B | 41,028 | **8 s** |

A 25k-line company takes well under a minute. With an LLM add the API latency: 75 calls ≈ 3–6 min. Classification cost is O(profiles × leaves).

## LLM cost per 1,000 lines
The LLM sees at most `max_items` (1,500) unique items, 20 per call, so cost is capped per run and does not grow with file size.
- **Per call (estimate):** ≈ 1.3–1.5k input tokens (tree + 20 items) and ≈ 0.3k output tokens.
- **Company A (42k included lines):** 75 calls → ≈ **2.5k input + 0.5k output tokens per 1,000 lines**. For a 25k-line company: ≈ 3 calls per 1,000 lines → ≈ 4k input + 0.8k output tokens.
- **Money:** multiply by your model's price. **Local Ollama and free tiers cost 0; no key → 0.** (Token counts are estimates from prompt size, not billed usage.)
