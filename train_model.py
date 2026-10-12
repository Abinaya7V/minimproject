"""
Model Training & Evaluation Script
----------------------------------
Trains Logistic Regression (primary deployed model) and Random Forest (comparison)
on the CEAS_08 phishing dataset. Strips source/header artifacts to prevent corpus bias,
ensures zero test leakage, performs 5-fold cross-validation, and saves comprehensive
evaluation metrics and visualizations.
"""

import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from vectorizer import EmailVectorizer

# Artifact lines commonly found in CEAS / mailing list collections
_ARTIFACT_PATTERNS = [
    re.compile(r"^message-id:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^x-mailer:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^mime-version:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^content-type:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^return-path:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^received:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^from:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^to:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^subject:\s*.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"_{10,}.*?(?:mailing list|spambayes|mailman|listinfo).*?_{10,}", re.IGNORECASE | re.DOTALL),
    re.compile(r"http://mail\.python\.org/mailman/listinfo/\S+", re.IGNORECASE),
    re.compile(r"http://lists\.sourceforge\.net/lists/listinfo/\S+", re.IGNORECASE),
    re.compile(r"http://spambayes\.sourceforge\.net\S*", re.IGNORECASE),
]


def clean_email_artifacts(text: str) -> str:
    """Strips metadata header lines and mailing-list footers to prevent corpus bias."""
    if not isinstance(text, str):
        return ""
    cleaned = text
    for pat in _ARTIFACT_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    return cleaned.strip()


def load_dataset(csv_path: str):
    """Loads CSV, validates columns, maps binary labels, and cleans duplicates/artifacts."""
    print(f"[INFO] Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check required columns
    body_col = "body" if "body" in df.columns else ("text" if "text" in df.columns else None)
    if body_col is None or "label" not in df.columns:
        raise ValueError(f"Dataset must contain 'body' (or 'text') and 'label' columns. Found: {df.columns.tolist()}")

    total_raw = len(df)
    print(f"[INFO] Raw dataset rows loaded: {total_raw:,}")

    # Standardize labels: 1 = Phishing, 0 = Legitimate
    def normalize_label(val):
        s = str(val).strip().lower()
        if s in ["1", "1.0", "phishing", "phish", "spam"]:
            return 1
        elif s in ["0", "0.0", "legitimate", "legit", "ham"]:
            return 0
        return None

    df["binary_label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)

    # Optional subject inclusion
    if "subject" in df.columns:
        df["subject_clean"] = df["subject"].fillna("").astype(str).str.strip()
        df["combined_raw"] = df["subject_clean"] + "\n\n" + df[body_col].fillna("").astype(str)
    else:
        df["combined_raw"] = df[body_col].fillna("").astype(str)

    # Clean header artifacts
    print("[INFO] Sanitizing dataset: stripping metadata headers and list footers...")
    df["clean_text"] = df["combined_raw"].apply(clean_email_artifacts)

    # Drop empty texts
    df = df[df["clean_text"].str.strip() != ""]
    empty_dropped = total_raw - len(df)
    if empty_dropped > 0:
        print(f"[INFO] Dropped {empty_dropped:,} empty or header-only rows.")

    # Deduplicate text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    dupes_dropped = before_dedup - len(df)
    if dupes_dropped > 0:
        print(f"[INFO] Dropped {dupes_dropped:,} duplicate email bodies.")

    # Source / Sender inspection
    phish_count = int((df["binary_label"] == 1).sum())
    legit_count = int((df["binary_label"] == 0).sum())
    total_valid = len(df)

    print("\n" + "=" * 60)
    print("DATASET PROFILE & CLASS BALANCE")
    print("=" * 60)
    print(f"Total Valid Emails:      {total_valid:,}")
    print(f"Phishing Emails (1):     {phish_count:,} ({phish_count / total_valid * 100:.2f}%)")
    print(f"Legitimate Emails (0):   {legit_count:,} ({legit_count / total_valid * 100:.2f}%)")

    # Source bias analysis
    if "sender" in df.columns:
        sender_counts = df["sender"].dropna().value_counts()
        print(f"Unique senders detected: {len(sender_counts):,}")
        top_senders = sender_counts.head(5)
        print("Top 5 senders by frequency:")
        for s, count in top_senders.items():
            print(f"  - {s[:40]}: {count} occurrences")

    # Source bias warning check
    print("\n[SOURCE BIAS CHECK]")
    print("The CEAS_08 benchmark integrates legitimate public discussions (e.g. open source listservs)")
    print("with real-world phishing feeds. To prevent the classifier from memorizing listserv routing")
    print("tags or protocol headers instead of semantic phishing indicators, header stripping was applied.")
    print("=" * 60 + "\n")

    return df["clean_text"].tolist(), df["binary_label"].tolist(), phish_count, legit_count, total_valid


def plot_metrics(lr_cm, rf_cm, metrics_summary, results_dir="results"):
    """Saves confusion matrix PNGs and a side-by-side performance bar chart."""
    os.makedirs(results_dir, exist_ok=True)

    # 1. Confusion Matrix - Logistic Regression
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=["Legitimate", "Phishing"])
    disp_lr.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix - Logistic Regression\n(Deployed Model)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    lr_cm_path = os.path.join(results_dir, "confusion_matrix_lr.png")
    plt.savefig(lr_cm_path)
    plt.close()

    # 2. Confusion Matrix - Random Forest
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=["Legitimate", "Phishing"])
    disp_rf.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix - Random Forest\n(Comparison Model)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    rf_cm_path = os.path.join(results_dir, "confusion_matrix_rf.png")
    plt.savefig(rf_cm_path)
    plt.close()

    # 3. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_scores = [
        metrics_summary["lr"]["accuracy"],
        metrics_summary["lr"]["precision"],
        metrics_summary["lr"]["recall"],
        metrics_summary["lr"]["f1"]
    ]
    rf_scores = [
        metrics_summary["rf"]["accuracy"],
        metrics_summary["rf"]["precision"],
        metrics_summary["rf"]["recall"],
        metrics_summary["rf"]["f1"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, lr_scores, width, label="Logistic Regression", color="#2563eb")
    rects2 = ax.bar(x + width/2, rf_scores, width, label="Random Forest", color="#059669")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Model Performance Comparison (Phishing Detection)", fontsize=12, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"[INFO] Visualizations saved to '{results_dir}/'")


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dataset Discovery
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        dataset_path = sys.argv[1]
    elif os.path.exists("data/CEAS_08.csv"):
        dataset_path = "data/CEAS_08.csv"
    elif os.path.exists("sample_data.csv"):
        dataset_path = "sample_data.csv"
    else:
        raise FileNotFoundError("Could not find dataset at 'data/CEAS_08.csv' or 'sample_data.csv'.")

    # 2. Load & Clean
    X_raw, y_raw, phish_count, legit_count, total_count = load_dataset(dataset_path)

    # 3. Stratified Train/Test Split (80/20) with random_state=42
    print("[INFO] Splitting dataset into 80% train and 20% test (stratified, random_state=42)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=0.20,
        random_state=42,
        stratify=y_raw
    )
    print(f"Train samples: {len(X_train_raw):,} | Test samples: {len(X_test_raw):,}")

    # 4. Feature Extraction & Vectorization (Fit on train ONLY - no leakage)
    max_features = 20000 if len(X_raw) > 50000 else 10000
    print(f"[INFO] Vectorizing text using TF-IDF (max_features={max_features}, n-grams=(1,2))...")
    vectorizer = EmailVectorizer(max_features=max_features, min_df=2, ngram_range=(1, 2))
    
    print("[INFO] Fitting TF-IDF vocabulary strictly on training set...")
    X_train = vectorizer.fit_transform(X_train_raw)

    print("[INFO] Transforming holdout test set...")
    X_test = vectorizer.transform(X_test_raw)

    # 5. Model 1: Logistic Regression (Balanced, Deployed Model)
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION (DEPLOYED MODEL)")
    print("=" * 60)
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    print(f"Accuracy:  {acc_lr:.4f}")
    print(f"Precision: {prec_lr:.4f}")
    print(f"Recall:    {rec_lr:.4f}")
    print(f"F1-Score:  {f1_lr:.4f}")
    print("Confusion Matrix:\n", cm_lr)

    # 5-Fold Stratified Cross-Validation for Logistic Regression
    print("\n[INFO] Running 5-fold Stratified Cross-Validation (F1-score) on Training Set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=skf, scoring="f1")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"5-Fold CV F1 Scores: {[round(float(s), 4) for s in cv_scores]}")
    print(f"5-Fold CV Mean F1:   {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 6. Model 2: Random Forest (Comparison Only)
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST (COMPARISON MODEL)")
    print("=" * 60)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=30
    )
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    print(f"Accuracy:  {acc_rf:.4f}")
    print(f"Precision: {prec_rf:.4f}")
    print(f"Recall:    {rec_rf:.4f}")
    print(f"F1-Score:  {f1_rf:.4f}")
    print("Confusion Matrix:\n", cm_rf)

    # 7. Extract Top 15 Sanity Check Words
    feature_names = vectorizer.get_feature_names()
    coefs = lr_model.coef_[0]

    top_phish_idx = np.argsort(coefs)[::-1][:15]
    top_legit_idx = np.argsort(coefs)[:15]

    top_phish_words = [(feature_names[i], round(float(coefs[i]), 3)) for i in top_phish_idx]
    top_legit_words = [(feature_names[i], round(float(coefs[i]), 3)) for i in top_legit_idx]

    print("\n" + "=" * 60)
    print("TOP 15 WORDS PUSHING TOWARD PHISHING (+ Weight)")
    print("=" * 60)
    for rank, (w, val) in enumerate(top_phish_words, 1):
        print(f" {rank:2d}. {w:25s} (+{val:.3f})")

    print("\n" + "=" * 60)
    print("TOP 15 WORDS PUSHING TOWARD LEGITIMATE (- Weight)")
    print("=" * 60)
    for rank, (w, val) in enumerate(top_legit_words, 1):
        print(f" {rank:2d}. {w:25s} ({val:.3f})")

    # 8. Save Metrics Report to results/metrics.txt
    metrics_summary = {
        "lr": {"accuracy": acc_lr, "precision": prec_lr, "recall": rec_lr, "f1": f1_lr, "cm": cm_lr},
        "rf": {"accuracy": acc_rf, "precision": prec_rf, "recall": rec_rf, "f1": f1_rf, "cm": cm_rf}
    }

    metrics_text_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_text_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AI-POWERED PHISHING DETECTION: MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset Path:             {dataset_path}\n")
        f.write(f"Total Valid Samples:      {total_count:,}\n")
        f.write(f"Phishing Samples (1):     {phish_count:,} ({phish_count / total_count * 100:.2f}%)\n")
        f.write(f"Legitimate Samples (0):   {legit_count:,} ({legit_count / total_count * 100:.2f}%)\n")
        f.write(f"Train / Test Split:       80% Train ({len(X_train_raw):,}), 20% Holdout Test ({len(X_test_raw):,})\n")
        f.write(f"Vocabulary Features:      {len(feature_names):,} terms\n\n")

        f.write("-" * 70 + "\n")
        f.write("TEST SET PERFORMANCE COMPARISON (Phishing Class = 1)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Metric':<20} | {'Logistic Regression':<22} | {'Random Forest':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Accuracy':<20} | {acc_lr:<22.4f} | {acc_rf:<20.4f}\n")
        f.write(f"{'Precision':<20} | {prec_lr:<22.4f} | {prec_rf:<20.4f}\n")
        f.write(f"{'Recall':<20} | {rec_lr:<22.4f} | {rec_rf:<20.4f}\n")
        f.write(f"{'F1-Score':<20} | {f1_lr:<22.4f} | {f1_rf:<20.4f}\n")
        f.write(f"{'5-Fold CV Mean F1':<20} | {cv_mean:<22.4f} | {'N/A (Comparison Only)':<20}\n")
        f.write(f"{'5-Fold CV Std':<20} | {cv_std:<22.4f} | {'N/A':<20}\n")
        f.write("-" * 70 + "\n\n")

        f.write("CONFUSION MATRIX - LOGISTIC REGRESSION (DEPLOYED):\n")
        f.write(f"  [TN: {cm_lr[0][0]:5d}  FP: {cm_lr[0][1]:5d}]\n")
        f.write(f"  [FN: {cm_lr[1][0]:5d}  TP: {cm_lr[1][1]:5d}]\n\n")

        f.write("CONFUSION MATRIX - RANDOM FOREST (COMPARISON):\n")
        f.write(f"  [TN: {cm_rf[0][0]:5d}  FP: {cm_rf[0][1]:5d}]\n")
        f.write(f"  [FN: {cm_rf[1][0]:5d}  TP: {cm_rf[1][1]:5d}]\n\n")

        f.write("TOP 15 PHISHING INDICATOR WORDS (+ Logistic Regression Weight):\n")
        for rank, (w, val) in enumerate(top_phish_words, 1):
            f.write(f"  {rank:2d}. {w:<25} (+{val:.3f})\n")
        f.write("\n")

        f.write("TOP 15 LEGITIMATE INDICATOR WORDS (- Logistic Regression Weight):\n")
        for rank, (w, val) in enumerate(top_legit_words, 1):
            f.write(f"  {rank:2d}. {w:<25} ({val:.3f})\n")
        f.write("\n")
        f.write("DEPLOYMENT NOTE: Logistic Regression is deployed to production in 'phishing_model.pkl'\n")
        f.write("to preserve linear log-odds feature attribution for Chrome Extension explainability.\n")

    # 9. Plot Charts
    plot_metrics(cm_lr, cm_rf, metrics_summary, results_dir=results_dir)

    # 10. Save Deployed Artifacts
    print("\n[INFO] Saving deployed Logistic Regression model and vectorizer...")
    joblib.dump(lr_model, "phishing_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("[SUCCESS] Exported phishing_model.pkl and vectorizer.pkl successfully.")


if __name__ == "__main__":
    main()
