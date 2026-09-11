# NLP Preprocessing Module

**Project:** AI-Powered Email Phishing Detection Chrome Extension
**Module:** NLP Preprocessing (TF-IDF, tokenizing)
**Status:** Complete and tested — standalone, no dependency on other modules

## What this module does

Converts raw email text (HTML, headers, noise and all) into clean, numeric
features that a downstream ML model can classify as phishing or legitimate.

```
raw email text
      │
      ▼
strip HTML tags
      │
      ▼
lowercase + normalize whitespace
      │
      ▼
tag URLs / email addresses  →  URLTOKEN / EMAILTOKEN
      │                         (presence of a link is itself a phishing
      │                          signal, so we mark it instead of deleting it)
      ▼
remove punctuation & numbers
      │
      ▼
tokenize → remove stopwords → lemmatize
      │
      ▼
clean_tokens  ──►  TF-IDF vectorizer  ──►  numeric feature matrix
```

## Files

| File | Purpose |
|---|---|
| `preprocessor.py` | Core cleaning + tokenizing pipeline |
| `vectorizer.py` | Wraps the preprocessor, produces TF-IDF vectors |
| `test_preprocessor.py` | Unit tests (6/6 passing) |
| `requirements.txt` | Dependencies |

## How to run it

```bash
pip install -r requirements.txt
python preprocessor.py      # demo: cleans one sample phishing email
python vectorizer.py        # demo: TF-IDF on 2 phishing + 2 legit emails
python test_preprocessor.py # runs unit tests
```

## Why this design

- **URLs/emails are tagged, not deleted** — their presence is signal for
  the Feature Extraction module later, not noise to throw away.
- **A few "stopwords" are deliberately kept** (`now`, `urgent`, `click`,
  `verify`, `immediately`, `account`) because standard NLTK stopword lists
  treat them as filler, but in phishing detection they're some of the
  strongest indicators.
- **Standalone by design** — this module takes plain text in and returns
  clean text/vectors out. It doesn't touch Gmail, the Flask API, or the
  ML model, so it could be built, tested, and demoed completely
  independently of the rest of the team's work, which is exactly what a
  Review-1 "at least one completed module" deliverable needs.

## Demo output (from `vectorizer.py`)

Phishing emails surface terms like `verify`, `urgent`, `locked`, `password`.
Legitimate emails surface terms like `team`, `lunch`, `sync`, `thanks`.
This separation is exactly what lets the ML Classification module (Person C)
tell the two classes apart later.

## Next steps (future modules, not part of this deliverable)

- Feature Extraction module (Person B) will add non-text features: sender
  domain reputation, number of URLs, presence of attachments, etc.
- ML Classification module (Person C) will train Logistic Regression /
  Random Forest on the TF-IDF vectors produced here.
