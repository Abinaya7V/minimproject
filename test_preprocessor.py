"""
Basic unit tests for the NLP Preprocessing module.
Run with: python3 -m pytest test_preprocessor.py -v
(or just: python3 test_preprocessor.py)
"""

from preprocessor import (
    strip_html,
    normalize_text,
    remove_urls_and_emails,
    remove_punctuation_and_numbers,
    tokenize_and_lemmatize,
    preprocess_email,
)


def test_strip_html():
    assert strip_html("<b>Hello</b> world").strip() == "Hello world"
    assert strip_html("") == ""


def test_normalize_text():
    assert normalize_text("  HELLO   World  ") == "hello world"


def test_remove_urls_and_emails():
    text = "Visit http://fake-bank.com or email admin@fake.com now"
    out = remove_urls_and_emails(text)
    assert "URLTOKEN" in out
    assert "EMAILTOKEN" in out
    assert "fake-bank.com" not in out


def test_remove_punctuation_and_numbers():
    out = remove_punctuation_and_numbers("Act now!!! Within 24 hours.")
    assert "!" not in out
    assert "24" not in out


def test_tokenize_and_lemmatize():
    tokens = tokenize_and_lemmatize("verifying accounts immediately")
    assert "verifying" in tokens or "verify" in tokens  # lemmatized form
    assert "and" not in tokens  # stopword removed


def test_full_pipeline_keeps_signal_words():
    sample = "URGENT!! Click here now to verify your account: http://scam.fake"
    cleaned = preprocess_email(sample)
    for signal_word in ["urgent", "click", "verify", "account", "URLTOKEN"]:
        assert signal_word in cleaned, f"expected '{signal_word}' in: {cleaned}"


def run_all():
    tests = [
        test_strip_html,
        test_normalize_text,
        test_remove_urls_and_emails,
        test_remove_punctuation_and_numbers,
        test_tokenize_and_lemmatize,
        test_full_pipeline_keeps_signal_words,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
