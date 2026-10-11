
# AI Phishing Detection Demo Web App

This is a live, working Flask web application that visually ties together the entire backend pipeline for your college project demo:

- **NLP Preprocessing** (`preprocessor.py`, `vectorizer.py`): Cleans text and extracts TF-IDF vectors.
- **Feature Extraction** (`feature_extractor.py`): Uses advanced heuristics to check sender domains, find hidden malicious links, and count keywords.
- **ML Classification** (`predict.py`, `train_model.py`): Uses a Logistic Regression model to make the final prediction based on the TF-IDF representation.

## Setup & Run Instructions (Under 2 Minutes)

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure ML Model Exists:**
   If you haven't yet trained the model, do so to generate the `.pkl` files:
   ```bash
   python train_model.py
   ```

3. **Start the Web App:**
   ```bash
   python app.py
   ```

4. **View the Live Demo:**
   Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

## Demo Tips
The UI includes pre-filled sample buttons on the main screen so you can demonstrate the app live without manually copy/pasting text! Try clicking "Try Sample Phishing 1" and then hit "Analyze Email" to instantly show off the advanced URL and domain mismatch detection!
