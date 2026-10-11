from flask import Flask, render_template, request
import os
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    email_text = request.form.get('email_text', '')
    sender_email = request.form.get('sender_email', '')
    subject = request.form.get('subject', '')
    
    try:
        from predict import predict_email
        from feature_extractor import extract_features, feature_vector_summary
        
        # 1. Get ML prediction from our saved model
        result = predict_email(email_text)
        
        # 2. Get advanced feature heuristics (URLs, Sender mismatch, Keywords)
        features = extract_features(email_text, sender_email=sender_email, subject=subject)
        advanced_reasons = feature_vector_summary(features)
        
        # 3. Merge reasons (deduplicate basic ML reasons with Advanced Feature reasons)
        ml_reasons = result.get('reasons', [])
        combined_reasons = list(set(ml_reasons + advanced_reasons))
        
        # If no reasons were triggered but it's phishing, add a generic fallback
        if result['prediction'] == 'Phishing' and not combined_reasons:
             combined_reasons = ["Email text pattern matches known phishing characteristics."]
        elif result['prediction'] == 'Legitimate' and not combined_reasons:
             combined_reasons = ["No suspicious indicators found. Text pattern matches legitimate communications."]
             
        result['reasons'] = combined_reasons
        
        return render_template('result.html', result=result, email_text=email_text)
    
    except FileNotFoundError as e:
        return render_template('result.html', error=str(e))
    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")

if __name__ == '__main__':
    app.run(debug=True)
