import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract
from email.utils import parseaddr

# Phishing heuristics lists
URGENT_KEYWORDS = [
    "urgent", "verify your account", "act now", "suspended", "click immediately",
    "limited time", "action required", "immediate action", "important update",
    "security alert", "account restricted", "verify identity", "unauthorized login",
    "update your information", "validate your account", "expires soon",
    "final notice", "confirm your account", "attention required", "account closure"
]

FINANCIAL_KEYWORDS = [
    "password", "otp", "bank account", "credit card", "wire transfer",
    "invoice", "payment", "transaction", "billing", "ssn", "social security",
    "routing number", "account number", "swift code", "deposit", "withdrawal"
]

KNOWN_BRANDS = {
    "paypal": "paypal.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "google": "google.com",
    "amazon": "amazon.com",
    "bank of america": "bankofamerica.com",
    "chase": "chase.com",
    "wells fargo": "wellsfargo.com",
    "citi": "citi.com",
    "netflix": "netflix.com",
    "facebook": "facebook.com",
    "linkedin": "linkedin.com",
    "dhl": "dhl.com",
    "fedex": "fedex.com",
    "ups": "ups.com"
}

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "adf.ly", "shorte.st"
]

def extract_features(email_text, sender_email=None, subject=None):
    """
    Extracts phishing features from an email text and metadata.
    
    Args:
        email_text (str): The body of the email (HTML or plain text).
        sender_email (str): The sender's email address, optionally with display name.
        subject (str): The subject line of the email.
        
    Returns:
        dict: A dictionary of extracted features.
    """
    features = {
        "num_urls": 0,
        "suspicious_url_flag": False,
        "sender_domain": None,
        "sender_domain_mismatch": False,
        "urgent_keyword_count": 0,
        "financial_keyword_count": 0,
        "has_attachment_mention": False,
        "exclamation_count": 0,
        "all_caps_word_count": 0
    }
    
    # 1. URL extraction & analysis
    soup = BeautifulSoup(email_text, 'html.parser')
    links = soup.find_all('a')
    
    # Find URLs in plain text as well
    url_pattern = re.compile(r'(?:https?://|www\.)[^\s<>"]+')
    text_urls = url_pattern.findall(email_text)
    
    total_urls = []
    
    for a in links:
        href = a.get('href', '')
        if href:
            total_urls.append(href)
            # Check mismatched display text (e.g. href is scam.com but display is paypal.com)
            display_text = a.text.strip()
            if display_text.startswith('http') and display_text != href:
                features["suspicious_url_flag"] = True
                
    for url in text_urls:
        if url not in total_urls:
            total_urls.append(url)
            
    features["num_urls"] = len(total_urls)
    
    for url in total_urls:
        # Normalize protocol if missing for urlparse
        if not url.startswith('http'):
            url = 'http://' + url
            
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        
        # Check shorteners
        if any(shortener in netloc for shortener in SHORTENERS):
            features["suspicious_url_flag"] = True
            
        # Check IP address instead of domain (IPv4 format)
        if re.match(r'^\d{1,3}(\.\d{1,3}){3}(:\d+)?$', netloc):
            features["suspicious_url_flag"] = True
            
    # 2. Sender domain & brand mismatch
    if sender_email:
        name, address = parseaddr(sender_email)
        if address:
            ext = tldextract.extract(address)
            # Reconstruct domain.suffix
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            features["sender_domain"] = domain
            
            # Check for brand mismatch
            name_lower = name.lower()
            for brand, true_domain in KNOWN_BRANDS.items():
                if brand in name_lower and domain != true_domain:
                    features["sender_domain_mismatch"] = True
                    
    # 3. Keyword analysis (case insensitive)
    text_lower = email_text.lower()
    if subject:
        text_lower += " " + subject.lower()
        
    for kw in URGENT_KEYWORDS:
        features["urgent_keyword_count"] += text_lower.count(kw)
        
    for kw in FINANCIAL_KEYWORDS:
        features["financial_keyword_count"] += text_lower.count(kw)
        
    # 4. Attachment mention
    attachment_phrases = ["attachment", "attached file", "attached document", "find attached"]
    if any(phrase in text_lower for phrase in attachment_phrases):
        features["has_attachment_mention"] = True
        
    # 5. Exclamation count
    features["exclamation_count"] = email_text.count("!")
    
    # 6. All caps word count (exclude acronyms <= 3 chars like OTP)
    # Match words composed only of alphabet letters
    words = re.findall(r'\b[A-Za-z]+\b', email_text)
    caps_words = [w for w in words if w.isupper() and len(w) > 3]
    features["all_caps_word_count"] = len(caps_words)
    
    return features


def feature_vector_summary(features_dict):
    """
    Converts features dictionary into a list of human-readable reasons.
    Suitable for displaying in the Chrome Extension UI.
    """
    reasons = []
    
    if features_dict.get("num_urls", 0) > 0:
        reasons.append(f'{features_dict["num_urls"]} URL(s) detected')
        
    if features_dict.get("suspicious_url_flag"):
        reasons.append("Suspicious URL format detected (URL shortener, IP address, or hidden link)")
        
    if features_dict.get("sender_domain"):
        # We don't necessarily flag every domain, but we can list it for info.
        pass
        
    if features_dict.get("sender_domain_mismatch"):
        reasons.append("Sender domain does not match claimed brand display name")
        
    if features_dict.get("urgent_keyword_count", 0) > 0:
        reasons.append(f'{features_dict["urgent_keyword_count"]} urgent keyword(s) found')
        
    if features_dict.get("financial_keyword_count", 0) > 0:
        reasons.append(f'{features_dict["financial_keyword_count"]} financial keyword(s) found')
        
    if features_dict.get("has_attachment_mention"):
        reasons.append("Email mentions an attachment")
        
    if features_dict.get("exclamation_count", 0) > 0:
        reasons.append(f'{features_dict["exclamation_count"]} exclamation mark(s) used')
        
    if features_dict.get("all_caps_word_count", 0) > 0:
        reasons.append(f'{features_dict["all_caps_word_count"]} fully capitalized word(s) found')
        
    return reasons
