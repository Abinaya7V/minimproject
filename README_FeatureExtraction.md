# Feature Extraction Module

**Project:** AI-Powered Email Phishing Detection Chrome Extension
**Module:** Feature Extraction
**Status:** Complete and tested

## What this module does

Takes raw or preprocessed email text and sender metadata, and extracts a structured dictionary of features that serve as indicators for phishing. It complements the NLP Preprocessing module by handling metadata (sender domain), HTML structure (suspicious URLs), and specific keyword counts.

```text
raw email + metadata
      │
      ▼
extract URLs & check for suspicious patterns (shorteners, IP addresses, mismatched text)
      │
      ▼
analyze sender domain & brand mismatch (e.g., claiming to be "PayPal" from a non-PayPal domain)
      │
      ▼
count urgent and financial keywords
      │
      ▼
flag attachments, exclamation marks, and ALL CAPS words
      │
      ▼
features_dict  ──►  feature_vector_summary (human readable reasons)
```

## Files

| File | Purpose |
|---|---|
| `feature_extractor.py` | Core logic to extract features and generate summaries |
| `test_feature_extractor.py` | Unit tests to ensure features are properly extracted |
| `README_FeatureExtraction.md` | This documentation file |

## How to run it

```bash
pip install -r requirements.txt
python test_feature_extractor.py # runs unit tests
```

## Why this design

- **Plugs into the existing pipeline** — it processes the text and supplements the TF-IDF features with heuristics that TF-IDF alone can't capture (like mismatched sender domains or disguised links).
- **Human-readable summaries** — the `feature_vector_summary` function translates the dictionary into plain English "reasons", which is exactly what an end user needs to see in the Chrome Extension UI to understand *why* an email was flagged.
- **Lightweight dependencies** — uses `tldextract` and `BeautifulSoup` for precise parsing without bloating the project.
