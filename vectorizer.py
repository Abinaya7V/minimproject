"""
TF-IDF Vectorization
---------------------
Takes cleaned text (output of preprocessor.py) and converts it into
numeric feature vectors using TF-IDF (Term Frequency - Inverse Document
Frequency). This is the bridge between "NLP Preprocessing" and the
"Feature Extraction" / "ML Classification" modules.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import preprocess_email


class EmailVectorizer:
    def __init__(self, max_features: int = 5000, min_df: int = 2, ngram_range=(1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self._fitted = False

    def fit(self, raw_emails: list):
        """Clean + fit TF-IDF vocabulary on training set only."""
        cleaned = [preprocess_email(email) for email in raw_emails]
        self.vectorizer.fit(cleaned)
        self._fitted = True
        return self

    def fit_transform(self, raw_emails: list):
        """Clean + fit TF-IDF vocabulary on a list of raw email texts, return matrix."""
        cleaned = [preprocess_email(email) for email in raw_emails]
        matrix = self.vectorizer.fit_transform(cleaned)
        self._fitted = True
        return matrix

    def transform(self, raw_emails: list):
        """Clean + vectorize new emails using an already-fitted vocabulary."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform or fit first to build the vocabulary.")
        cleaned = [preprocess_email(email) for email in raw_emails]
        return self.vectorizer.transform(cleaned)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
