"""
Test API script to evaluate retrained model across 4 email scenarios:
1. Obvious phishing email
2. Normal work email
3. Marketing newsletter with several links
4. Empty body (expects 400 Bad Request)
"""

from app import app

client = app.test_client()

test_cases = [
    {
        "name": "1. Obvious Phishing Email",
        "payload": {
            "subject": "URGENT: Your account has been suspended! Verify immediately",
            "sender": "security-alert@paypal-update-center.com",
            "body": "Dear customer, we detected unauthorized login activity. Please verify your password and account credentials immediately at http://scam-verify-login.biz/auth or your account will be permanently locked."
        },
        "expect_status": 200
    },
    {
        "name": "2. Normal Work Email",
        "payload": {
            "subject": "Sync meeting notes and next steps",
            "sender": "priya.sharma@company.org",
            "body": "Hi team, thanks for joining the morning sync. Attached are the action items from our discussion. Let me know if you have any questions or feedback before our Friday review."
        },
        "expect_status": 200
    },
    {
        "name": "3. Newsletter with Several Links",
        "payload": {
            "subject": "Tech Weekly Digest: Python 3.12 Updates and AI Trends",
            "sender": "newsletter@techdigest-daily.com",
            "body": "Welcome to this week's digest! Read our featured article on modern web architecture at https://techdigest-daily.com/articles/web-arch. Check out open source tooling at https://github.com/trending and subscribe to our podcast at https://techdigest-daily.com/podcast. You can manage your preferences or unsubscribe at https://techdigest-daily.com/preferences."
        },
        "expect_status": 200
    },
    {
        "name": "4. Empty Body (Validation Test)",
        "payload": {
            "subject": "No body provided",
            "sender": "test@example.com",
            "body": "   "
        },
        "expect_status": 400
    }
]

print("=" * 70)
print("FLASK API TEST CLIENT RESULTS")
print("=" * 70)

for tc in test_cases:
    res = client.post('/api/analyze', json=tc["payload"])
    assert res.status_code == tc["expect_status"], f"Expected {tc['expect_status']} but got {res.status_code}"
    data = res.get_json()

    print(f"\nScenario: {tc['name']}")
    print(f"Status Code: {res.status_code}")
    if res.status_code == 200:
        print(f"  label:                {data.get('label')}")
        print(f"  confidence:           {data.get('confidence')}%")
        print(f"  phishing_probability: {data.get('phishing_probability')}%")
        print(f"  risk_level:           {data.get('risk_level')}")
        print(f"  reasons count:        {len(data.get('reasons', []))}")
        print(f"  recommendations count:{len(data.get('recommendations', []))}")
    else:
        print(f"  error message:        {data.get('error')}")

print("\n" + "=" * 70)
print("ALL 4 API TEST SCENARIOS PASSED SUCCESSFULLY!")
print("=" * 70)
