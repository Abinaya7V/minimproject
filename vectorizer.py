"""
TF-IDF Vectorization
---------------------
TF=Tearm Frequency, IDF=Inverse Document Frequency. 
This module takes the cleaned text output from preprocessor.py and converts it into numeric feature
vectors using the TF-IDF algorithm. This is the bridge between "NLP Preprocessing"
and the "Feature Extraction" / "ML Classification" modules that come later.
Takes cleaned text (output of preprocessor.py) and converts it into
numeric feature vectors using TF-IDF (Term Frequency - Inverse Document
Frequency). This is the bridge between "NLP Preprocessing" and the
"Feature Extraction" / "ML Classification" modules that come later.

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import preprocess_email


class EmailVectorizer:
    def __init__(self, max_features: int = 500):
        # max_features caps vocabulary size -> keeps model fast & light,
        # good enough for a mini-project scale dataset
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit_transform(self, raw_emails: list):
        """Clean + fit TF-IDF vocabulary on a list of raw email texts, return matrix."""
        cleaned = [preprocess_email(email) for email in raw_emails]
        matrix = self.vectorizer.fit_transform(cleaned)
        self._fitted = True
        return matrix

    def transform(self, raw_emails: list):
        """Clean + vectorize new emails using an already-fitted vocabulary."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first to build the vocabulary.")
        cleaned = [preprocess_email(email) for email in raw_emails]
        return self.vectorizer.transform(cleaned)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()


if __name__ == "__main__":
    # Small demo dataset: 2 phishing-style + 2 legitimate-style emails
    sample_emails = [
        # Phishing-style
        """Dear Customer, your account has been suspended! Verify your identity
        immediately by clicking here: http://secure-bank-login.fake/verify""",

        """URGENT: Your PayPal account will be locked in 24 hours. Click now
        to confirm your password at http://paypal-verify-account.fake""",

        # Legitimate-style
        """Hi team, attached is the agenda for tomorrow's 10am sync. Let me
        know if you want to add anything before then. Thanks, Priya""",

        """Hey, just checking if you're free for lunch on Friday. Let me know
        what time works for you!""",
    ]
    labels = ["phishing", "phishing", "legit", "legit"]

    ev = EmailVectorizer(max_features=50)
    matrix = ev.fit_transform(sample_emails)

    print(f"TF-IDF matrix shape: {matrix.shape}  (emails x vocabulary terms)\n")
    print("Vocabulary learned:\n", sorted(ev.get_feature_names()), "\n")

    for i, email_label in enumerate(labels):
        row = matrix[i].toarray()[0]
        top_indices = row.argsort()[::-1][:5]
        top_terms = [(ev.get_feature_names()[j], round(row[j], 3))
                     for j in top_indices if row[j] > 0]
        print(f"Email {i+1} [{email_label}] top TF-IDF terms: {top_terms}")
