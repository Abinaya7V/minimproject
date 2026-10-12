"""
AI-Powered Email Phishing Detection - Flask Backend & API Server
-----------------------------------------------------------------
Provides:
1. Web Demo UI (HTML templates for presentation and manual testing)
2. JSON REST API (for Chrome Extension integration and automated clients)

Privacy Notice:
No email content (sender, subject, body) is stored, persisted to database,
or recorded in server log files.
"""

import os
import re
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from predict import (
    load_model_artifacts,
    predict_email,
    get_top_contributing_words,
    get_recommendations,
    calculate_risk_level
)
from feature_extractor import extract_features, feature_vector_summary
from preprocessor import _lemmatizer, _cached_lemmatize

# Initialize Flask Application
app = Flask(__name__)

# -------------------------------------------------------------------------
# CORS Configuration:
# Allow requests from Google Chrome Extension origins (chrome-extension://*)
# as well as local development hosts (localhost / 127.0.0.1 on any port).
# -------------------------------------------------------------------------
CORS(app, resources={
    r"/api/*": {
        "origins": [
            re.compile(r"^chrome-extension://.*"),
            re.compile(r"^http://localhost(:\d+)?$"),
            re.compile(r"^http://127\.0\.0\.1(:\d+)?$")
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# -------------------------------------------------------------------------
# Startup Model & Resource Pre-loading:
# Load ML model, vectorizer, and pre-warm NLP resources ONCE into memory
# at server startup so subsequent requests respond in milliseconds.
# -------------------------------------------------------------------------
try:
    load_model_artifacts()
    # Trigger a lightweight warm-up lookup
    _cached_lemmatize("account")
    print("[SUCCESS] Machine learning model and NLP resources pre-loaded at startup.")
except Exception as e:
    print(f"[WARNING] Model pre-load deferred (run train_model.py if files are missing): {e}")


# =========================================================================
# WEB DEMO ROUTES (Maintains compatibility with HTML demo interface)
# =========================================================================

@app.route('/')
def index():
    """Renders the interactive web demo homepage."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Form-based endpoint used by the interactive web demo.
    Renders templates/result.html with classification results and heuristics.
    """
    email_text = request.form.get('email_text', '')
    sender_email = request.form.get('sender_email', '')
    subject = request.form.get('subject', '')

    try:
        # 1. Evaluate prediction using cached model
        result = predict_email(email_text)

        # 2. Extract heuristic features (URLs, domain mismatch, formatting)
        features = extract_features(email_text, sender_email=sender_email, subject=subject)
        advanced_reasons = feature_vector_summary(features)

        # 3. Merge ML explainability reasons with heuristic reasons
        ml_reasons = result.get('reasons', [])
        combined_reasons = list(dict.fromkeys(ml_reasons + advanced_reasons))

        if result['prediction'] == 'Phishing' and not combined_reasons:
            combined_reasons = ["Email text pattern matches known phishing characteristics."]
        elif result['prediction'] == 'Legitimate' and not combined_reasons:
            combined_reasons = ["No suspicious indicators found. Text pattern matches legitimate communications."]

        result['reasons'] = combined_reasons

        # Render result template without logging or persisting the email text
        return render_template('result.html', result=result, email_text=email_text)

    except FileNotFoundError as e:
        return render_template('result.html', error=str(e))
    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")


# =========================================================================
# JSON REST API ENDPOINTS (For Chrome Extension & External Clients)
# =========================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns: {"status": "ok"} with HTTP 200 status.
    """
    return jsonify({"status": "ok"}), 200


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    JSON API endpoint for the Chrome Extension.
    
    Expected JSON input:
        {
            "subject": str,   # Optional subject line
            "sender": str,    # Optional sender header (e.g. "Name <user@domain.com>")
            "body": str       # Required email body
        }
        
    Output JSON (200 OK):
        {
            "label": "Phishing" | "Legitimate",
            "confidence": float (0-100),
            "risk_level": "Low" | "Medium" | "High",
            "reasons": [str],
            "recommendations": [str]
        }
    """
    # 1. Validate that the request contains JSON
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type. Request payload must be JSON with 'Content-Type: application/json'."
        }), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Malformed JSON payload. Expected a JSON object."
        }), 400

    # 2. Validate input fields: 'body' is required and must not be empty
    body = data.get("body")
    if body is None or not isinstance(body, str) or not body.strip():
        return jsonify({
            "error": "Missing or empty required field: 'body'. Email content must be provided."
        }), 400

    subject = data.get("subject", "")
    sender = data.get("sender", "")

    if not isinstance(subject, str):
        subject = str(subject or "")
    if not isinstance(sender, str):
        sender = str(sender or "")

    try:
        # Combine subject and body for classification if subject is present
        analysis_text = f"{subject}\n\n{body}".strip() if subject.strip() else body

        # 3. Perform prediction (predict_email internally computes X and extracts TF-IDF reasons)
        prediction_result = predict_email(analysis_text)
        tfidf_reasons = prediction_result.get("reasons", [])

        # 4. Extract additional feature heuristics (sender mismatch, URLs, etc.)
        heuristic_features = extract_features(body, sender_email=sender, subject=subject)
        heuristic_reasons = feature_vector_summary(heuristic_features)

        # Combine TF-IDF reasons with heuristic reasons (preserving order and uniqueness)
        merged_reasons = list(dict.fromkeys(tfidf_reasons + heuristic_reasons))
        if not merged_reasons:
            if prediction_result["label"] == "Phishing":
                merged_reasons = ["Email text patterns match known phishing characteristics."]
            else:
                merged_reasons = ["No suspicious indicators detected in the message."]

        # 5. Format API response matching project specifications
        # Privacy: We return only analysis metrics; email content is NEVER logged or saved.
        response_payload = {
            "label": prediction_result["label"],
            "confidence": prediction_result["confidence_pct"],  # float between 0 and 100
            "phishing_probability": prediction_result["phishing_probability"],  # float between 0 and 100
            "risk_level": prediction_result["risk_level"],
            "reasons": merged_reasons,
            "recommendations": prediction_result["recommendations"]
        }

        return jsonify(response_payload), 200

    except FileNotFoundError as e:
        return jsonify({
            "error": "Model or vectorizer file not found on server.",
            "details": str(e)
        }), 500
    except Exception as e:
        # Never log email content in traceback/logs
        return jsonify({
            "error": "Internal classification error occurred.",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    # Default Flask development server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
