"""
NLP Preprocessing Module
-------------------------
Part of: AI-Powered Email Phishing Detection Chrome Extension
Module: NLP Preprocessing (TF-IDF, tokenizing)

Purpose: Takes raw email text (which may contain HTML, headers, URLs, noise)
and converts it into clean, tokenized, lemmatized text ready for feature
extraction / ML classification.

Pipeline:
  raw_text -> strip HTML -> lowercase -> remove URLs/emails/numbers/punct
           -> tokenize -> remove stopwords -> lemmatize -> clean_tokens
"""

import re
import string
from functools import lru_cache
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and pre-warm WordNet so lookups don't block during requests
_lemmatizer = WordNetLemmatizer()
try:
    wordnet.ensure_loaded()
except Exception:
    pass

_stop_words = set(stopwords.words('english'))

# Keep a few words that matter a LOT for phishing detection even though
# they're normally treated as "stopwords" or filtered by generic cleaning.
# e.g. urgency/action words are signal, not noise.
_KEEP_WORDS = {"now", "urgent", "immediately", "verify", "click", "account"}
_stop_words = _stop_words - _KEEP_WORDS

# Pre-compiled regex patterns for maximum execution speed
_RE_WHITESPACE = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RE_DIGITS = re.compile(r"\d+")
_PUNCT_TRANSLATE = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=10000)
def _cached_lemmatize(token: str) -> str:
    """Caches lemmatized forms of common English words to eliminate redundant lookups."""
    return _lemmatizer.lemmatize(token)


def strip_html(text: str) -> str:
    """Remove HTML tags from email body (emails are often HTML-formatted)."""
    if not text:
        return ""
    # Only invoke BeautifulSoup if HTML tags are actually present
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        raw = soup.get_text(separator=" ")
    else:
        raw = text
    return _RE_WHITESPACE.sub(" ", raw).strip()


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    return _RE_WHITESPACE.sub(" ", text).strip()


def remove_urls_and_emails(text: str, replace_with_tag: bool = True) -> str:
    """
    Replace URLs and email addresses with placeholder tokens instead of
    deleting them outright. WHY: presence of a URL/email address is itself
    a strong phishing signal that the Feature Extraction module will use
    later — so we tag it rather than erase the evidence.
    """
    if not text:
        return ""
    if replace_with_tag:
        text = _RE_URL.sub(" URLTOKEN ", text)
        text = _RE_EMAIL.sub(" EMAILTOKEN ", text)
    else:
        text = _RE_URL.sub(" ", text)
        text = _RE_EMAIL.sub(" ", text)
    return text


def remove_punctuation_and_numbers(text: str) -> str:
    """Strip punctuation and standalone numbers (keeps URLTOKEN/EMAILTOKEN intact)."""
    if not text:
        return ""
    text = _RE_DIGITS.sub(" ", text)
    text = text.translate(_PUNCT_TRANSLATE)
    return _RE_WHITESPACE.sub(" ", text).strip()


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, drop stopwords, lemmatize each token to its base form."""
    if not text:
        return []
    tokens = word_tokenize(text)
    cleaned = [
        _cached_lemmatize(tok)
        for tok in tokens
        if tok not in _stop_words and len(tok) > 1
    ]
    return cleaned


def preprocess_email(raw_text: str, return_string: bool = True):
    """
    Full pipeline entry point.

    Args:
        raw_text: raw email body (may include HTML)
        return_string: if True, returns space-joined string (good for TF-IDF);
                        if False, returns list of tokens

    Returns:
        Cleaned text as string or list of tokens
    """
    text = strip_html(raw_text)
    text = normalize_text(text)
    text = remove_urls_and_emails(text)
    text = remove_punctuation_and_numbers(text)
    tokens = tokenize_and_lemmatize(text)

    if return_string:
        return " ".join(tokens)
    return tokens


if __name__ == "__main__":
    sample = """
    <html><body>
    Dear Customer,<br>
    Your account has been <b>suspended</b>! Please verify your identity
    immediately by clicking here: http://secure-bank-login.fake/verify
    or contact support@totally-legit-bank.com within 24 hours.
    </body></html>
    """
    print("RAW:\n", sample)
    print("\nCLEANED STRING:\n", preprocess_email(sample))
    print("\nCLEANED TOKENS:\n", preprocess_email(sample, return_string=False))
