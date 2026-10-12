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


def get_top_contributing_words(email_text: str, model=None, vectorizer=None, label: str = "Phishing", top_n: int = 3) -> list:
    """
    Extracts the top TF-IDF words contributing to the prediction using Logistic
    Regression coefficients.
    
    Formula:
        contribution_i = X[i] * coef_[i]
        
    - For Phishing: positive contributions push towards class 1 (Phishing).
    - For Legitimate: negative contributions push towards class 0 (Legitimate).
    
    Written as an isolated function so the Feature Extraction module or other
    explainability modules can easily import and extend it.
    
    Args:
        email_text: The raw email body/subject text.
        model: Fitted LogisticRegression model (optional, uses cached by default).
        vectorizer: Fitted EmailVectorizer instance (optional, uses cached by default).
        label: Predicted label ("Phishing" or "Legitimate").
        top_n: Maximum number of contributing words to return.
        
    Returns:
        list[str]: Human-readable reasons explaining the prediction.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    try:
        # Transform the incoming text into TF-IDF representation
        X = vectorizer.transform([email_text])
        row = X.toarray()[0]
        feature_names = vectorizer.get_feature_names()
        coefs = model.coef_[0]

        contributions = []
        for idx in range(len(row)):
            if row[idx] > 0:
                word = feature_names[idx]
                # Contribution is TF-IDF term frequency * model coefficient
                weight = float(row[idx] * coefs[idx])
                contributions.append((word, weight))

        if label == "Phishing":
            # For phishing, sort by highest positive contribution
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
            # For legitimate, sort by most negative contribution (pushing toward class 0)
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
        # Graceful fallback in case of transformation edge cases
        return [f"Text pattern analysis completed ({str(e)})."]


def get_recommendations(risk_level: str, label: str = "Phishing") -> list:
    """
    Returns safety guidance tailored to the assessed risk level.
    """
    if label == "Phishing":
        if risk_level == "High":
            return [
                "Do not click on any links or download attached files.",
                "Do not provide credentials, passwords, or personal identity numbers.",
                "Verify the sender's identity through an independent, official channel.",
                "Report this email immediately to your organization's IT or security team."
            ]
        elif risk_level == "Medium":
            return [
                "Exercise caution: hover over links to inspect destination URLs before clicking.",
                "Double-check the sender address for subtle misspellings or spoofed domains.",
                "Never enter login credentials on unverified third-party pages.",
                "If the message demands urgent action, confirm validity with the sender directly."
            ]
        else:  # Low risk phishing
            return [
                "Proceed with caution and verify sender authenticity.",
                "Inspect any embedded URLs to ensure they point to legitimate official domains."
            ]
    else:
        return [
            "Email appears legitimate, but always observe standard security precautions.",
            "Never share sensitive passwords or verification codes over email."
        ]


def calculate_risk_level(label: str, confidence_pct: float) -> str:
    """
    Computes risk level according to project requirements:
    - For Phishing predictions:
        <40% -> Low
        40-70% -> Medium
        >70% -> High
    - For Legitimate predictions:
        Low
    """
    if label == "Phishing":
        if confidence_pct > 70.0:
            return "High"
        elif confidence_pct >= 40.0:
            return "Medium"
        else:
            return "Low"
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
        dict: Contains label, confidence (both 0-1 and 0-100), risk_level, reasons, recommendations.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    # Preprocess and vectorize the input text
    X = vectorizer.transform([email_text])

    # Predict probabilities: prob[0] = Legitimate, prob[1] = Phishing
    prob = model.predict_proba(X)[0]
    is_phishing = bool(model.predict(X)[0] == 1)

    # Calculate confidence values
    confidence_raw = float(prob[1] if is_phishing else prob[0])
    confidence_pct = round(confidence_raw * 100.0, 2)
    label = "Phishing" if is_phishing else "Legitimate"

    # Determine risk level based on specified thresholds
    risk_level = calculate_risk_level(label, confidence_pct)

    # Derive explainability reasons using Logistic Regression coefficients
    reasons = get_top_contributing_words(email_text, model, vectorizer, label=label, top_n=3)

    # Generate actionable security recommendations
    recommendations = get_recommendations(risk_level, label=label)

    return {
        "label": label,                          # API requirement standard
        "prediction": label,                     # Web demo template backwards-compatibility
        "confidence": round(confidence_raw, 4),  # Web demo template (0.0 - 1.0)
        "confidence_pct": confidence_pct,        # JSON API requirement (0.0 - 100.0)
        "risk_level": risk_level,
        "reasons": reasons,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    # Test script directly
    test_phishing = "URGENT: Verify your bank account immediately or it will be suspended."
    test_legit = "Hi team, are we still on for the 10am sync tomorrow? Thanks."

    print("--- Test Phishing ---")
    res_phish = predict_email(test_phishing)
    for k, v in res_phish.items():
        print(f"  {k}: {v}")

    print("\n--- Test Legitimate ---")
    res_legit = predict_email(test_legit)
    for k, v in res_legit.items():
        print(f"  {k}: {v}")
