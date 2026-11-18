# Procurement Spend Intelligence System

**Config-Driven Data Engineering & Machine Learning Pipeline**

A configurable Python-based system designed to transform raw procurement and purchase data into structured, deduplicated, and classified spend intelligence.

The system is designed to process data from multiple sources, identify valid procurement spend, standardise suppliers and items, and classify transactions into company-specific category taxonomies — including previously unseen category trees.

---

## 1. Project Overview

Organisations often maintain procurement information across multiple ERP systems, invoice platforms, and financial databases. Differences in supplier names, product descriptions, currencies, transaction formats, and duplicate records make spend analysis challenging.

The Procurement Spend Intelligence System addresses these challenges through an automated and configuration-driven processing pipeline.

**Core objective:** Convert unstructured and inconsistent procurement records into reliable, standardised, and categorised spend data.

## 2. Key Capabilities

- **Multi-Source Data Ingestion:** Process procurement records from different CSV sources using configurable column mappings.
- **Data Standardisation:** Handle inconsistent date formats, numeric representations, missing values, and descriptions.
- **Spend Identification:** Identify eligible procurement transactions and exclude cancelled, unapproved, duplicate, and non-procurement records.
- **Net Spend Calculation:** Calculate spend excluding applicable taxes and convert currencies into the reporting currency.
- **Supplier Deduplication:** Identify and standardise multiple representations of the same supplier.
- **Item Deduplication:** Identify equivalent products and services using available identifiers and descriptions.
- **Automated Classification:** Assign procurement records to the appropriate category hierarchy.
- **Confidence Scoring:** Generate classification confidence scores between 0 and 1.
- **Configuration-Driven Processing:** Adapt to new companies and category trees through configuration rather than company-specific classification code.

## 3. System Architecture

```text
       RAW PROCUREMENT DATA
                 |
                 v
        DATA INGESTION
                 |
                 v
       DATA STANDARDISATION
                 |
                 v
       SPEND IDENTIFICATION
                 |
                 v
       SPEND CALCULATION
                 |
                 v
       RECORD MATCHING
                 |
                 v
       SUPPLIER & ITEM
        DEDUPLICATION
                 |
                 v
       CATEGORY CLASSIFICATION
                 |
                 v
        CONFIDENCE SCORING
                 |
                 v
          FINAL OUTPUT
                 |
                 v
          STRUCTURED CSV
```

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Pandas | Data processing and manipulation |
| NumPy | Numerical operations |
| PyYAML | Configuration management |
| RapidFuzz | Fuzzy matching and entity resolution |
| Scikit-learn | Machine learning and evaluation |
| Sentence Transformers | Semantic similarity and classification |
| Python-dateutil | Date parsing |
| tqdm | Processing progress |
| pytest | Automated testing |
| Groq API | Optional LLM-assisted processing |
| uv | Python dependency and environment management |

The design includes a local processing fallback so that the pipeline does not depend entirely on an external LLM API.

## 5. Project Structure

```text
cantill-spend-tool/
│
├── configs/
│   ├── company_A.yaml
│   ├── company_B.yaml
│   └── company_template.yaml
│
├── data/                       # Local confidential datasets
│
├── spendkit/
│   ├── __init__.py
│   ├── config.py
│   ├── parsing.py
│   ├── ingest.py
│   ├── amounts.py
│   ├── exclusions.py
│   ├── matching.py
│   ├── dedup.py
│   ├── classify.py
│   └── run.py
│
├── tests/
│   └── test_parsing.py
│
├── submission/
│
├── README.md
├── APPROACH.md
├── DECISIONS.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── .gitignore
```

## 6. Installation & Setup

### Prerequisites

- Python 3.11
- Git
- uv (recommended)

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd cantill-spend-tool
```

### Install dependencies using uv

```bash
uv sync
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### Configure environment variables

For optional Groq API integration, configure your API key through an environment variable.

```env
GROQ_API_KEY=your_api_key
```

Never commit API keys or confidential credentials to GitHub.

## 7. Configuration

The pipeline uses YAML configuration files to define company-specific processing requirements.

Example:

```yaml
company:
  name: Company_A
  reporting_currency: ZAR

processing:
  default_vat_rate: 0.15
  approved_statuses:
    - Approved
    - Paid

sources:
  - name: erp
    file: erp_goods_receipts.csv
    id_prefix: "ERP:"
    amount_basis: net

  - name: invoice_tool
    file: invoice_tool_export.csv
    id_prefix: "TOOL:"
    amount_basis: gross
```

This is an illustrative configuration. Actual column mappings, date formats, exclusion rules, and taxonomy settings are defined in the company-specific configuration.

## 8. Execution

Run the processing pipeline using a single command:

```bash
uv run python -m spendkit.run --config configs/company_A.yaml --out output/company_A.csv
```

For Company B:

```bash
uv run python -m spendkit.run --config configs/company_B.yaml --out output/company_B.csv
```

The execution interface is designed to support company-specific configurations without modifying the core processing logic.

## 9. Output Schema

The pipeline follows a standardised output structure.

| Column | Description |
|---|---|
| record_id | Unique source record identifier |
| include | Whether the record contributes to spend (1/0) |
| exclusion_reason | Reason for exclusion |
| supplier_key | Standardised supplier identifier |
| item_key | Standardised item identifier |
| spend | Net spend in reporting currency |
| l1 | Category level 1 |
| l2 | Category level 2 |
| l3 | Category level 3 |
| l4 | Category level 4 |
| confidence | Classification confidence score (0–1) |

Every input record is intended to have a corresponding output record, including excluded transactions.

## 10. Testing & Evaluation

The project includes automated tests for core data parsing and standardisation functionality.

Run tests using:

```bash
uv run pytest -v
```

Evaluation focuses on:

- Spend inclusion and exclusion accuracy.
- Duplicate transaction identification.
- Supplier and item deduplication quality.
- Category classification accuracy.
- Performance on previously unseen category taxonomies.
- Processing time and API usage.

A key evaluation requirement is the ability to classify procurement records without relying on labelled examples for the target company.

## 11. Design Principles

1. **Configuration over Hardcoding:** Company-specific rules should reside in configuration files.
2. **Modular Architecture:** Each processing stage should remain independently testable.
3. **Data Integrity:** Preserve source records and maintain traceability.
4. **Scalability:** Avoid unnecessary pairwise comparisons and use efficient matching strategies.
5. **Explainability:** Maintain clear exclusion reasons and classification confidence.
6. **Resilience:** Support local processing when external API services are unavailable.

## 12. Data Confidentiality

All challenge datasets are confidential and are intentionally excluded from this repository.

The local `data/` directory is excluded through `.gitignore`.

No original task datasets, confidential records, or API credentials should be committed.

## 13. Project Information

**Project:** Procurement Spend Intelligence System  
**Organisation:** Cantill.ai (Straightops Global)  
**Challenge:** Data & ML Intern — Round 1  
**Category:** Data Engineering | Machine Learning | NLP | Spend Analytics  
**Language:** Python 3.11

---

*Built as a modular, configuration-driven approach to automated procurement spend intelligence.*
