import pandas as pd
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from vectorizer import EmailVectorizer
import os

def generate_sample_data():
    """Generates a small synthetic dataset for demo purposes."""
    data = {
        "text": [
            "URGENT: Your account is suspended. Verify now at http://scam.com",
            "Please confirm your password for the wire transfer.",
            "Click immediately to claim your prize! Limited time offer.",
            "Action required: Security alert for your bank account.",
            "Verify your identity to prevent account closure.",
            "Final notice: Invoice attached. Please act now.",
            "Unauthorized login detected. Click here to secure your account.",
            "Update your billing information or your Netflix will be suspended.",
            "Your PayPal account is restricted. Verify your account.",
            "Important update regarding your social security number.",
            "Hi team, are we still on for the 10am sync tomorrow?",
            "Can you review the attached document when you have a moment?",
            "Let's grab lunch on Friday. I know a great spot.",
            "Thanks for the update. I will check it out and get back to you.",
            "Here are the meeting notes from yesterday.",
            "Just checking in on the project status. Any blockers?",
            "Don't forget to submit your timesheets by EOD.",
            "Happy birthday! Hope you have a great day.",
            "The code has been deployed to staging. Please test.",
            "Can we reschedule our 1-on-1 to next week?"
        ],
        "label": [
            "phishing", "phishing", "phishing", "phishing", "phishing",
            "phishing", "phishing", "phishing", "phishing", "phishing",
            "legitimate", "legitimate", "legitimate", "legitimate", "legitimate",
            "legitimate", "legitimate", "legitimate", "legitimate", "legitimate"
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv("sample_data.csv", index=False)
    print("Created synthetic sample_data.csv with 20 examples.")

def main():
    if not os.path.exists("sample_data.csv"):
        generate_sample_data()

    # Load dataset
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)
    
    if "text" not in df.columns or "label" not in df.columns:
        print("Error: CSV must contain 'text' and 'label' columns.")
        sys.exit(1)
        
    X_raw = df["text"].fillna("").tolist()
    y_raw = df["label"].fillna("legitimate").tolist()
    
    # Map labels to binary (1 for phishing, 0 for legitimate)
    y = [1 if str(label).lower().strip() == "phishing" else 0 for label in y_raw]
    
    print("Preprocessing and vectorizing text...")
    ev = EmailVectorizer(max_features=500)
    X = ev.fit_transform(X_raw)
    
    print("Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("\nSaving model and vectorizer...")
    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(ev, "vectorizer.pkl")
    print("Saved to phishing_model.pkl and vectorizer.pkl.")

if __name__ == "__main__":
    main()
