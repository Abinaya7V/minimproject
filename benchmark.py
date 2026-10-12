import time
from predict import predict_email, get_top_contributing_words, load_model_artifacts
from feature_extractor import extract_features

# Generate ~15,000 characters of realistic email text with 10 links
fake_paragraphs = [
    "Dear Valued Customer, We have detected unusual login activity on your account from an unrecognized device. ",
    "Please verify your credentials immediately to avoid service interruption and account suspension. ",
    "You can review your recent transactions and account settings by visiting http://secure-update-portal.com/login?id=12345. ",
    "Important notice: Your monthly statement is ready for download at https://my-bank-reports.org/statement.pdf. ",
    "If you did not authorize this change, please contact customer support immediately at support@bank-security.net. ",
    "Failure to take action within 24 hours may result in permanent account termination and forfeiture of funds. ",
    "Please visit https://verify-identity-service.com/auth to confirm your social security number and billing information. ",
    "Click here http://reward-claims.biz/claim?user=customer to claim your promotional gift card bonus. ",
    "Check our privacy policy at https://legitcompany.com/privacy and terms of service at https://legitcompany.com/terms. ",
    "Thank you for being a valued customer. Our security team works 24/7 to protect your financial safety. "
]

# Repeat paragraphs to reach ~15,000 characters
long_body = ("\n\n".join(fake_paragraphs) * 12)[:15000]
subject = "URGENT: Verify your account immediately or access will be suspended"
sender = "Security Team <security@bank-alert-update.com>"

print(f"Benchmark input length: {len(long_body)} characters")

# Ensure models are loaded
load_model_artifacts()

# Measure predict_email
t0 = time.perf_counter()
pred = predict_email(f"{subject}\n\n{long_body}")
t_predict = time.perf_counter() - t0

# Measure get_top_contributing_words
t0 = time.perf_counter()
reasons = get_top_contributing_words(f"{subject}\n\n{long_body}", label=pred["label"])
t_reasons = time.perf_counter() - t0

# Measure extract_features
t0 = time.perf_counter()
features = extract_features(long_body, sender_email=sender, subject=subject)
t_features = time.perf_counter() - t0

total_time = t_predict + t_reasons + t_features

print("\n--- TIMING RESULTS ---")
print(f"predict_email:              {t_predict:.4f} s")
print(f"get_top_contributing_words: {t_reasons:.4f} s")
print(f"extract_features:           {t_features:.4f} s")
print(f"Total time:                 {total_time:.4f} s")
