import unittest
from feature_extractor import extract_features, feature_vector_summary

class TestFeatureExtractor(unittest.TestCase):
    
    def test_phishing_email_urgent_and_mismatch(self):
        email_text = "URGENT: Verify your account immediately! We noticed suspicious activity."
        sender = '"PayPal Support" <admin@scammer.com>'
        subject = "Action Required"
        
        features = extract_features(email_text, sender_email=sender, subject=subject)
        
        self.assertTrue(features["sender_domain_mismatch"])
        self.assertEqual(features["sender_domain"], "scammer.com")
        self.assertGreater(features["urgent_keyword_count"], 0)
        self.assertEqual(features["all_caps_word_count"], 1) # URGENT
        self.assertEqual(features["exclamation_count"], 1)

    def test_phishing_suspicious_url_mismatched_text(self):
        email_text = 'Please login here: <a href="http://192.168.1.1/login">http://paypal.com/login</a>'
        sender = "someone@example.com"
        
        features = extract_features(email_text, sender_email=sender)
        
        self.assertTrue(features["suspicious_url_flag"])
        self.assertEqual(features["num_urls"], 2)

    def test_legitimate_email(self):
        email_text = "Hi team, let's have lunch tomorrow at 12. Thanks!"
        sender = "colleague@company.com"
        
        features = extract_features(email_text, sender_email=sender)
        
        self.assertEqual(features["num_urls"], 0)
        self.assertFalse(features["suspicious_url_flag"])
        self.assertFalse(features["sender_domain_mismatch"])
        self.assertEqual(features["urgent_keyword_count"], 0)
        self.assertEqual(features["financial_keyword_count"], 0)

    def test_attachment_and_financial(self):
        email_text = "Please find the attached document for the invoice and wire transfer."
        sender = "vendor@supplier.com"
        
        features = extract_features(email_text, sender_email=sender)
        
        self.assertTrue(features["has_attachment_mention"])
        self.assertGreater(features["financial_keyword_count"], 0)
        
    def test_url_shortener(self):
        email_text = "Check out this link: https://bit.ly/12345"
        
        features = extract_features(email_text)
        
        self.assertTrue(features["suspicious_url_flag"])
        self.assertEqual(features["num_urls"], 1)
        
    def test_feature_vector_summary(self):
        features = {
            "num_urls": 2,
            "suspicious_url_flag": True,
            "sender_domain": "scammer.com",
            "sender_domain_mismatch": True,
            "urgent_keyword_count": 3,
            "financial_keyword_count": 0,
            "has_attachment_mention": False,
            "exclamation_count": 0,
            "all_caps_word_count": 0
        }
        summary = feature_vector_summary(features)
        self.assertEqual(len(summary), 4)
        self.assertIn("2 URL(s) detected", summary)
        self.assertIn("Suspicious URL format detected (URL shortener, IP address, or hidden link)", summary)
        self.assertIn("Sender domain does not match claimed brand display name", summary)

if __name__ == "__main__":
    unittest.main()
