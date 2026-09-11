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
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize once (avoids reloading on every call)
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))

# Keep a few words that matter a LOT for phishing detection even though
# they're normally treated as "stopwords" or filtered by generic cleaning.
# e.g. urgency/action words are signal, not noise.
_KEEP_WORDS = {"now", "urgent", "immediately", "verify", "click", "account"}
_stop_words = _stop_words - _KEEP_WORDS


def strip_html(text: str) -> str:
    """Remove HTML tags from email body (emails are often HTML-formatted)."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    raw = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", raw).strip()


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_urls_and_emails(text: str, replace_with_tag: bool = True) -> str:
    """
    Replace URLs and email addresses with placeholder tokens instead of
    deleting them outright. WHY: presence of a URL/email address is itself
    a strong phishing signal that the Feature Extraction module will use
    later — so we tag it rather than erase the evidence.
    """
    url_pattern = r"(https?://\S+|www\.\S+)"
    email_pattern = r"\S+@\S+\.\S+"

    if replace_with_tag:
        text = re.sub(url_pattern, " URLTOKEN ", text)
        text = re.sub(email_pattern, " EMAILTOKEN ", text)
    else:
        text = re.sub(url_pattern, " ", text)
        text = re.sub(email_pattern, " ", text)
    return text


def remove_punctuation_and_numbers(text: str) -> str:
    """Strip punctuation and standalone numbers (keeps URLTOKEN/EMAILTOKEN intact)."""
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, drop stopwords, lemmatize each token to its base form."""
    tokens = word_tokenize(text)
    cleaned = [
        _lemmatizer.lemmatize(tok)
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
