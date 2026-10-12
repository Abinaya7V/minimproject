/**
 * AI Email Phishing Detector - Popup Controller
 * --------------------------------------------
 * Connects the popup UI to the active tab's content script and background service worker.
 */

// UI Elements
const statusPill = document.getElementById('status-pill');
const metaSubject = document.getElementById('meta-subject');
const metaSender = document.getElementById('meta-sender');
const metaUrlsCount = document.getElementById('meta-urls-count');
const analyzeBtn = document.getElementById('analyze-btn');
const spinner = document.getElementById('spinner');
const btnText = analyzeBtn.querySelector('.btn-text');

const errorCard = document.getElementById('error-card');
const errorMessage = document.getElementById('error-message');

const resultsCard = document.getElementById('results-card');
const resultLabel = document.getElementById('result-label');
const riskBadge = document.getElementById('risk-badge');
const confidenceNum = document.getElementById('confidence-num');
const meterFill = document.getElementById('meter-fill');
const reasonsList = document.getElementById('reasons-list');
const recommendationsList = document.getElementById('recommendations-list');

let currentEmailData = null;

/**
 * Checks if the local Flask backend is currently running
 */
async function checkBackendHealth() {
  try {
    const res = await fetch('http://127.0.0.1:5000/api/health', { method: 'GET' });
    if (res.ok) {
      const data = await res.json().catch(() => null);
      if (data && data.status === 'ok') {
        statusPill.textContent = 'Backend: Online';
        statusPill.className = 'status-indicator online';
        return true;
      }
    }
  } catch (e) {
    // Backend offline
  }

  statusPill.textContent = 'Backend: Offline';
  statusPill.className = 'status-indicator offline';
  return false;
}

/**
 * Displays error card and hides results
 */
function showError(msg) {
  errorMessage.textContent = msg;
  errorCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
}

/**
 * Hides error card
 */
function hideError() {
  errorCard.classList.add('hidden');
}

/**
 * Renders extraction metadata
 */
function setMetadata(subject, sender, urlsCount) {
  metaSubject.textContent = subject || '(No subject detected)';
  metaSender.textContent = sender || '(No sender detected)';
  metaUrlsCount.textContent = typeof urlsCount === 'number' ? urlsCount : '0';
}

/**
 * Renders model analysis results
 */
function renderResults(result) {
  hideError();
  resultsCard.classList.remove('hidden');

  const label = result.label || 'Unknown';
  const risk = (result.risk_level || 'Low').toLowerCase();
  const confidence = typeof result.confidence === 'number' ? result.confidence : 0;

  // Label styling
  resultLabel.textContent = label;
  resultLabel.className = `result-label ${label.toLowerCase()}`;

  // Risk badge
  riskBadge.textContent = `${result.risk_level || 'Low'} Risk`;
  riskBadge.className = `risk-badge ${risk}`;

  // Confidence & Progress Meter
  confidenceNum.textContent = `${Math.round(confidence)}%`;
  meterFill.style.width = `${Math.min(Math.max(confidence, 0), 100)}%`;
  meterFill.className = `meter-fill ${risk}`;

  // Reasons list
  reasonsList.innerHTML = '';
  const reasons = Array.isArray(result.reasons) && result.reasons.length > 0
    ? result.reasons
    : ['No specific risk keywords triggered.'];
  reasons.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    reasonsList.appendChild(li);
  });

  // Recommendations list
  recommendationsList.innerHTML = '';
  const recs = Array.isArray(result.recommendations) && result.recommendations.length > 0
    ? result.recommendations
    : ['Always verify sender legitimacy before clicking links or downloading attachments.'];
  recs.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    recommendationsList.appendChild(li);
  });
}

/**
 * Requests email metadata from the active tab's content script
 */
async function fetchEmailFromTab() {
  hideError();
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab || !tab.id) {
    showError("Could not identify the active browser tab.");
    analyzeBtn.disabled = true;
    return;
  }

  const url = tab.url || '';
  const isEmailClient = url.includes('mail.google.com') || url.includes('outlook.office.com') || url.includes('outlook.live.com');

  if (!isEmailClient) {
    setMetadata('-', '-', 0);
    showError("You are not currently on an email page. Please open Gmail (mail.google.com) or Outlook Web.");
    analyzeBtn.disabled = true;
    return;
  }

  // Request extraction from content.js
  chrome.tabs.sendMessage(tab.id, { type: 'GET_EMAIL' }, (response) => {
    if (chrome.runtime.lastError) {
      // Content script not loaded yet (e.g. extension loaded after the tab was opened)
      setMetadata('-', '-', 0);
      showError("Could not communicate with the email tab. Please reload your Gmail/Outlook tab once and try again.");
      analyzeBtn.disabled = true;
      return;
    }

    if (!response || response.error) {
      setMetadata(response ? response.subject : '-', response ? response.sender : '-', 0);
      showError(response ? response.error : "No email message appears to be currently selected.");
      analyzeBtn.disabled = true;
      return;
    }

    currentEmailData = response;
    setMetadata(response.subject, response.sender, response.urls ? response.urls.length : 0);
    analyzeBtn.disabled = false;
  });
}

/**
 * Handle "Analyze This Email" click
 */
async function onAnalyzeClick() {
  if (!currentEmailData || !currentEmailData.body) {
    showError("No email content available to analyze. Please open an email first.");
    return;
  }

  // Set loading state
  hideError();
  analyzeBtn.disabled = true;
  spinner.classList.remove('hidden');
  btnText.textContent = 'Analyzing...';

  // Send request via background service worker
  chrome.runtime.sendMessage({
    type: 'ANALYZE',
    payload: {
      subject: currentEmailData.subject || '',
      sender: currentEmailData.sender || '',
      body: currentEmailData.body || ''
    }
  }, async (response) => {
    // Reset button state
    analyzeBtn.disabled = false;
    spinner.classList.add('hidden');
    btnText.textContent = 'Analyze This Email';

    if (chrome.runtime.lastError) {
      showError(`Extension worker error: ${chrome.runtime.lastError.message}`);
      return;
    }

    if (!response || response.error) {
      showError(response ? response.error : 'Unknown analysis error occurred.');
      return;
    }

    if (response.success && response.result) {
      renderResults(response.result);

      // Instruct content script to show inline risk banner
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: 'SHOW_BANNER',
          payload: response.result
        }, () => {
          // Ignore if banner injection is dismissed or fails
          void chrome.runtime.lastError;
        });
      }
    }
  });
}

// Initial setup
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  fetchEmailFromTab();
  analyzeBtn.addEventListener('click', onAnalyzeClick);
});
