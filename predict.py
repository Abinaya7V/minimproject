"""
ML Prediction Module
---------------------
Part of: AI-Powered Email Phishing Detection Chrome Extension
Module: ML Classification & Inference

Provides model caching (loaded once at startup), inference,
explainability (top TF-IDF feature weights), and risk assessment.
"""

import os
import warnings
import joblib
import numpy as np

# Suppress sklearn version warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# Global cache so model and vectorizer are loaded ONCE into memory at startup
_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def load_model_artifacts(model_path: str = "phishing_model.pkl", vectorizer_path: str = "vectorizer.pkl"):
    """
    Loads the trained Logistic Regression model and TF-IDF vectorizer once into memory.
    Subsequent calls return the cached instances immediately.
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER

    if _CACHED_MODEL is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_MODEL, _CACHED_VECTORIZER

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Model file ('{model_path}') or vectorizer file ('{vectorizer_path}') not found. "
            "Please ensure you run 'python train_model.py' first."
        )

    _CACHED_MODEL = joblib.load(model_path)
    _CACHED_VECTORIZER = joblib.load(vectorizer_path)
    return _CACHED_MODEL, _CACHED_VECTORIZER


def get_top_contributing_words(email_text: str = None, model=None, vectorizer=None, label: str = "Phishing", top_n: int = 3, X=None) -> list:
    """
    Extracts the top TF-IDF words contributing to the prediction using Logistic
    Regression coefficients.
    
    Formula:
        contribution_i = X[i] * coef_[i]
        
    - For Phishing: positive contributions push towards class 1 (Phishing).
    - For Legitimate: negative contributions push towards class 0 (Legitimate).
    
    Optimized to iterate directly over non-zero sparse matrix entries (X.indices / X.data)
    rather than dense array allocation, and optionally accepts an already-transformed vector X.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    try:
        # Reuse existing vectorized sparse row if passed; otherwise transform text
        if X is None:
            if email_text is None:
                return ["Analysis based on general text patterns."]
            X = vectorizer.transform([email_text])

        # Work directly with CSR sparse representation (zero-copy for non-zeros)
        csr_row = X.tocsr()
        indices = csr_row.indices
        data = csr_row.data
        feature_names = vectorizer.get_feature_names()
        coefs = model.coef_[0]

        contributions = []
        for idx, val in zip(indices, data):
            if val > 0:
                word = feature_names[idx]
                weight = float(val * coefs[idx])
                contributions.append((word, weight))

        if label == "Phishing":
            phish_words = [item for item in contributions if item[1] > 0]
            phish_words.sort(key=lambda x: x[1], reverse=True)
            
            reasons = [
                f"Suspicious keyword detected: '{word}' (model weight: {weight:+.2f})"
                for word, weight in phish_words[:top_n]
            ]
            if not reasons:
                reasons.append("Email text patterns match general phishing characteristics.")
            return reasons

        else:
            legit_words = [item for item in contributions if item[1] < 0]
            legit_words.sort(key=lambda x: x[1])  # most negative first
            
            reasons = [
                f"Standard communication term: '{word}' (model weight: {weight:+.2f})"
                for word, weight in legit_words[:top_n]
            ]
            if not reasons:
                reasons.append("No suspicious phishing keywords detected in message body.")
            return reasons

    except Exception as e:
        return [f"Text pattern analysis completed ({str(e)})."]


def get_recommendations(risk_level: str, label: str = "Phishing") -> list:
    """
    Returns safety guidance tailored to the assessed risk level.
    Provides cautious advice if an email is labeled Legitimate but has Medium risk.
    """
    if risk_level == "High":
        return [
            "Do not click on any links or download attached files.",
            "Do not provide credentials, passwords, or personal identity numbers.",
            "Verify the sender's identity through an independent, official channel.",
            "Report this email immediately to your organization's IT or security team."
        ]
    elif risk_level == "Medium":
        if label == "Phishing":
            return [
                "Exercise caution: hover over links to inspect destination URLs before clicking.",
                "Double-check the sender address for subtle misspellings or spoofed domains.",
                "Never enter login credentials on unverified third-party pages.",
                "If the message demands urgent action, confirm validity with the sender directly."
            ]
        else:
            return [
                "Exercise caution: email appears mostly normal, but moderate risk indicators were detected.",
                "Inspect links carefully before opening to confirm they point to official domains.",
                "Verify any unexpected requests for money, updates, or sensitive information."
            ]
    else:  # Low risk
        return [
            "Email appears safe, but maintain standard security vigilance.",
            "Never share sensitive passwords or verification codes over email."
        ]


def calculate_risk_level(phishing_probability: float) -> str:
    """
    Computes risk level from phishing_probability (0-100) for EVERY email:
    - below 30 = Low
    - 30 to 70 = Medium
    - above 70 = High
    """
    if phishing_probability > 70.0:
        return "High"
    elif phishing_probability >= 30.0:
        return "Medium"
    else:
        return "Low"


def predict_email(email_text: str, model=None, vectorizer=None) -> dict:
    """
    Predicts whether an email is phishing or legitimate.
    
    Args:
        email_text: Raw email text to classify.
        model: Optional pre-loaded model.
        vectorizer: Optional pre-loaded vectorizer.
        
    Returns:
        dict: Contains label, confidence (0-1 and 0-100), phishing_probability (0-100),
              risk_level, reasons, recommendations, and cached sparse row '_X'.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    # Preprocess and vectorize the input text (single transform)
    X = vectorizer.transform([email_text])

    # Predict probabilities: prob[0] = Legitimate, prob[1] = Phishing
    prob = model.predict_proba(X)[0]
    prob_legit = float(prob[0])
    prob_phish = float(prob[1])

    is_phishing = bool(model.predict(X)[0] == 1)

    # Confidence is probability of predicted class
    confidence_raw = prob_phish if is_phishing else prob_legit
    confidence_pct = round(confidence_raw * 100.0, 2)
    phishing_probability = round(prob_phish * 100.0, 2)
    label = "Phishing" if is_phishing else "Legitimate"

    # Risk level from phishing probability for EVERY email (<30 Low, 30-70 Medium, >70 High)
    risk_level = calculate_risk_level(phishing_probability)

    # Derive explainability reasons using sparse indices (no redundant transform)
    reasons = get_top_contributing_words(email_text=None, model=model, vectorizer=vectorizer, label=label, top_n=3, X=X)

    # Generate actionable security recommendations based on risk level and label
    recommendations = get_recommendations(risk_level, label=label)

    return {
        "label": label,                          # API requirement standard
        "prediction": label,                     # Web demo template backwards-compatibility
        "confidence": round(confidence_raw, 4),  # Web demo template (0.0 - 1.0)
        "confidence_pct": confidence_pct,        # JSON API requirement (0.0 - 100.0)
        "phishing_probability": phishing_probability, # Specific phishing risk percentage (0.0 - 100.0)
        "risk_level": risk_level,
        "reasons": reasons,
        "recommendations": recommendations,
        "_X": X                                  # Cached sparse vector
    }


if __name__ == "__main__":
    test_phishing = "URGENT: Verify your bank account immediately or it will be suspended."
    test_legit = "Hi team, are we still on for the 10am sync tomorrow? Thanks."

    print("--- Test Phishing ---")
    res_phish = predict_email(test_phishing)
    for k, v in res_phish.items():
        if k != "_X":
            print(f"  {k}: {v}")

    print("\n--- Test Legitimate ---")
    res_legit = predict_email(test_legit)
    for k, v in res_legit.items():
        if k != "_X":
            print(f"  {k}: {v}")
