import joblib
import os
import warnings

# Suppress sklearn version warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Simple keyword lists for generating inline reasons
URGENCY_WORDS = ["urgent", "verify", "suspended", "act now", "click immediately"]
FINANCIAL_WORDS = ["password", "otp", "bank account", "credit card"]

def predict_email(email_text):
    """
    Predicts whether an email is phishing or legitimate using the trained model.
    
    Args:
        email_text (str): The raw email text to classify.
        
    Returns:
        dict: A dictionary containing prediction, confidence score, risk level, and reasons.
    """
    if not os.path.exists("phishing_model.pkl") or not os.path.exists("vectorizer.pkl"):
        raise FileNotFoundError("Model or vectorizer not found. Please run train_model.py first.")
        
    # Load the trained model and the fitted vectorizer
    model = joblib.load("phishing_model.pkl")
    ev = joblib.load("vectorizer.pkl")
    
    # Preprocess and vectorize the input text
    X = ev.transform([email_text])
    
    # Get probabilities and prediction
    prob = model.predict_proba(X)[0]
    is_phishing = model.predict(X)[0] == 1
    
    # Confidence is the probability of the predicted class
    confidence = prob[1] if is_phishing else prob[0]
    prediction = "Phishing" if is_phishing else "Legitimate"
    
    # Determine risk level
    risk_level = "Low"
    if is_phishing:
        if confidence > 0.8:
            risk_level = "High"
        elif confidence >= 0.5:
            risk_level = "Medium"
            
    # Derive "reasons" using simple keyword checks inline
    reasons = []
    text_lower = email_text.lower()
    
    if is_phishing:
        # Check for urgency
        found_urgency = [w for w in URGENCY_WORDS if w in text_lower]
        if found_urgency:
            reasons.append(f"Contains urgent language: '{found_urgency[0]}'")
            
        # Check for financial terminology
        found_financial = [w for w in FINANCIAL_WORDS if w in text_lower]
        if found_financial:
            reasons.append(f"Contains financial/sensitive keywords: '{found_financial[0]}'")
            
        # Fallback generic reason if none found
        if not reasons:
            reasons.append("Email text pattern matches known phishing characteristics.")
            
    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "risk_level": risk_level,
        "reasons": reasons
    }

if __name__ == "__main__":
    # Quick demo to test prediction
    sample_phishing = "URGENT: Verify your bank account immediately or it will be suspended."
    print(f"Testing with: '{sample_phishing}'")
    try:
        result = predict_email(sample_phishing)
        print("\nPrediction Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
