/**
 * AI Email Phishing Detector - Background Service Worker
 * -------------------------------------------------------
 * Handles secure communication with the local Flask server at http://127.0.0.1:5000
 * Enforces payload trimming, 10s timeout via AbortController, and strict data privacy.
 */

const API_ANALYZE_URL = 'http://127.0.0.1:5000/api/analyze';
const REQUEST_TIMEOUT_MS = 30000;
const MAX_BODY_LENGTH = 20000; // Character limit to avoid huge payload overhead

/**
 * Handles the ANALYZE action by contacting the Flask backend.
 */
async function analyzeEmail(payload) {
  // Enforce privacy: never log subject, sender, or body to console or persistent storage
  const subject = typeof payload.subject === 'string' ? payload.subject : '';
  const sender = typeof payload.sender === 'string' ? payload.sender : '';
  const rawBody = typeof payload.body === 'string' ? payload.body : '';

  // Trim body if it exceeds maximum character limit
  const trimmedBody = rawBody.length > MAX_BODY_LENGTH
    ? rawBody.substring(0, MAX_BODY_LENGTH)
    : rawBody;

  if (!trimmedBody.trim()) {
    return {
      error: "Email body is empty or missing. Please select a valid email message to analyze."
    };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(API_ANALYZE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        subject: subject,
        sender: sender,
        body: trimmedBody
      }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const serverErrMsg = data && data.error ? data.error : `Server returned HTTP ${response.status}`;
      return {
        error: serverErrMsg
      };
    }

    if (!data) {
      return {
        error: "Malformed response received from local Flask backend."
      };
    }

    return {
      success: true,
      result: data
    };
  } catch (err) {
    clearTimeout(timeoutId);

    if (err.name === 'AbortError') {
      return {
        error: "Analysis request timed out after 10 seconds. The local backend took too long to respond."
      };
    }

    // Network error: typically server is offline
    return {
      error: "Cannot connect to local backend (http://127.0.0.1:5000). Please start the server: run 'python app.py' in your terminal."
    };
  }
}

// Listen for messages from popup.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE') {
    analyzeEmail(message.payload)
      .then(result => sendResponse(result))
      .catch(err => {
        sendResponse({
          error: `Unexpected extension error: ${err.message || err}`
        });
      });
    return true; // Keep asynchronous message channel open
  }
});
