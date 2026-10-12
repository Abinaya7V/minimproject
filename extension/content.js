/**
 * AI Email Phishing Detector - Content Script
 * -------------------------------------------
 * Injected into Gmail and Outlook Web tabs.
 * Extracts email metadata (subject, sender, body, URLs) and handles
 * inline UI banner injection.
 */

// ============================================================================
// CONFIG: Centralized DOM selectors for easy maintenance when Web UIs update
// ============================================================================
const CONFIG = {
  gmail: {
    // Subject line element
    subjectSelectors: [
      'h2.hP',
      'h2[data-thread-perm-id]',
      '[role="main"] h2',
      '.ha h2'
    ],
    // Sender element containing email attribute and display name
    senderSelectors: [
      'span.gD',
      'span[email]',
      '.gE span[email]',
      '[data-hovercard-id]'
    ],
    // Email body container (usually the last expanded message in a thread)
    bodySelectors: [
      'div.a3s.aiL',
      'div.a3s',
      '.adn.ads div.a3s',
      '[role="listitem"] div.a3s'
    ],
    // Insertion container for inline alert banner
    bannerContainerSelectors: [
      '.ha',
      'h2.hP',
      '[role="main"] .nH .if'
    ]
  },
  outlook: {
    // Subject line element
    subjectSelectors: [
      '[role="heading"][aria-level="2"]',
      'div[aria-label="Reading Pane"] h2',
      '.allowTextSelection[role="heading"]',
      '[data-automation-id="messageSubject"]'
    ],
    // Sender element
    senderSelectors: [
      '[data-testid="SenderPersona"]',
      '[aria-label*="From:"]',
      'span.personaEmail',
      '[data-automation-id="senderEmail"]',
      'div[role="main"] [data-hovercard-id]'
    ],
    // Email body container
    bodySelectors: [
      '[aria-label="Message body"]',
      'div#UniqueMessageBody',
      '[data-automation-id="readContractItem"]',
      'div.ReadingPane div[role="document"]'
    ],
    // Insertion container for inline alert banner
    bannerContainerSelectors: [
      '[role="heading"][aria-level="2"]',
      'div[aria-label="Reading Pane"]',
      '[data-automation-id="readContractItem"]'
    ]
  }
};

/**
 * Determine active provider based on hostname
 */
function getProvider() {
  const host = window.location.hostname.toLowerCase();
  if (host.includes('mail.google.com')) return 'gmail';
  if (host.includes('outlook.')) return 'outlook';
  return null;
}

/**
 * Helper to query the first matching element from an array of candidate selectors
 */
function queryFirst(selectors, root = document) {
  for (const selector of selectors) {
    try {
      const el = root.querySelector(selector);
      if (el) return el;
    } catch (e) {
      // Ignore invalid selector syntax if any
    }
  }
  return null;
}

/**
 * Helper to query all matching elements from an array of candidate selectors
 */
function queryAll(selectors, root = document) {
  for (const selector of selectors) {
    try {
      const elements = root.querySelectorAll(selector);
      if (elements && elements.length > 0) {
        return Array.from(elements);
      }
    } catch (e) {
      // Continue to next selector
    }
  }
  return [];
}

/**
 * Extracts sender display name and clean email address
 */
function extractSender(senderEl) {
  if (!senderEl) return '';

  // Check email attribute commonly used by Gmail (e.g. email="user@domain.com")
  const emailAttr = senderEl.getAttribute('email') || senderEl.getAttribute('data-hovercard-id') || '';
  const textName = (senderEl.innerText || senderEl.textContent || '').trim();

  if (emailAttr && textName && !textName.includes(emailAttr)) {
    return `${textName} <${emailAttr}>`;
  }
  return emailAttr || textName;
}

/**
 * Extract unique, valid HTTP/HTTPS URLs from the email body element
 */
function extractUrls(bodyEl) {
  if (!bodyEl) return [];
  const links = bodyEl.querySelectorAll('a[href]');
  const urls = new Set();

  links.forEach(a => {
    const href = a.getAttribute('href') || '';
    if (href.startsWith('http://') || href.startsWith('https://')) {
      urls.add(href);
    }
  });

  return Array.from(urls);
}

/**
 * Main email extraction routine.
 * Returns { subject, sender, body, urls } or { error: "..." }.
 */
function extractEmail() {
  const provider = getProvider();
  if (!provider) {
    return { error: 'Unsupported email web client. Please open Gmail or Outlook Web.' };
  }

  const selectors = CONFIG[provider];

  // 1. Extract Subject
  const subjectEl = queryFirst(selectors.subjectSelectors);
  const subject = subjectEl ? (subjectEl.innerText || subjectEl.textContent || '').trim() : '';

  // 2. Extract Sender
  const senderEl = queryFirst(selectors.senderSelectors);
  const sender = extractSender(senderEl);

  // 3. Extract Body (in Gmail threads, select the last visible/expanded message)
  const bodyCandidates = queryAll(selectors.bodySelectors);
  let bodyEl = null;

  if (bodyCandidates.length > 0) {
    // Pick the last candidate with non-trivial text
    for (let i = bodyCandidates.length - 1; i >= 0; i--) {
      const text = (bodyCandidates[i].innerText || '').trim();
      if (text.length > 0) {
        bodyEl = bodyCandidates[i];
        break;
      }
    }
    if (!bodyEl) bodyEl = bodyCandidates[bodyCandidates.length - 1];
  }

  if (!bodyEl) {
    return { error: 'No active email message open or message body not detected yet.' };
  }

  const body = (bodyEl.innerText || '').trim();
  if (!body) {
    return { error: 'Extracted email body is empty. Please select or expand an email.' };
  }

  // 4. Extract URLs
  const urls = extractUrls(bodyEl);

  return {
    subject,
    sender,
    body,
    urls,
    provider
  };
}

/**
 * Injects or updates an inline colored status banner above the email.
 */
function injectBanner(result) {
  removeBanner();

  if (!result || !result.risk_level) return;

  const provider = getProvider();
  if (!provider) return;

  const container = queryFirst(CONFIG[provider].bannerContainerSelectors);
  if (!container) return;

  const banner = document.createElement('div');
  banner.id = 'ai-phishing-detector-banner';

  const risk = (result.risk_level || 'Low').toLowerCase();
  const colors = {
    high: { bg: '#fee2e2', border: '#ef4444', text: '#991b1b', icon: '🚨' },
    medium: { bg: '#ffedd5', border: '#f97316', text: '#9a3412', icon: '⚠️' },
    low: { bg: '#dcfce7', border: '#22c55e', text: '#166534', icon: '🛡️' }
  };
  const theme = colors[risk] || colors.low;

  banner.style.cssText = `
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 10px 0;
    padding: 10px 14px;
    background-color: ${theme.bg};
    border-left: 5px solid ${theme.border};
    border-radius: 6px;
    color: ${theme.text};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    z-index: 9999;
  `;

  const confText = typeof result.confidence === 'number' ? ` (${result.confidence.toFixed(1)}% confidence)` : '';
  banner.innerHTML = `
    <div>
      <strong>${theme.icon} AI Phishing Guard:</strong>
      Predicted as <strong>${result.label || 'Unknown'}</strong> with <strong>${result.risk_level} Risk</strong>${confText}.
    </div>
    <button id="ai-banner-dismiss" style="
      background: none;
      border: none;
      color: ${theme.text};
      font-size: 16px;
      cursor: pointer;
      margin-left: 12px;
      padding: 0 4px;
      line-height: 1;
    " title="Dismiss">&times;</button>
  `;

  try {
    container.parentNode.insertBefore(banner, container);
    const dismissBtn = banner.querySelector('#ai-banner-dismiss');
    if (dismissBtn) {
      dismissBtn.addEventListener('click', () => removeBanner());
    }
  } catch (err) {
    console.warn('[AI Phishing Detector] Could not inject banner:', err);
  }
}

/**
 * Removes any existing alert banner
 */
function removeBanner() {
  const existing = document.getElementById('ai-phishing-detector-banner');
  if (existing && existing.parentNode) {
    existing.parentNode.removeChild(existing);
  }
}

// ============================================================================
// Message Listener: Communicates with popup.js
// ============================================================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_EMAIL') {
    const data = extractEmail();
    sendResponse(data);
    return true; // Keep channel open
  }

  if (message.type === 'SHOW_BANNER') {
    injectBanner(message.payload);
    sendResponse({ success: true });
    return true;
  }

  if (message.type === 'REMOVE_BANNER') {
    removeBanner();
    sendResponse({ success: true });
    return true;
  }
});

// ============================================================================
// Debounced MutationObserver to detect navigation between emails
// ============================================================================
let debounceTimer = null;
let lastSubject = '';

function onDomMutated() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    const provider = getProvider();
    if (!provider) return;

    const currentSubjectEl = queryFirst(CONFIG[provider].subjectSelectors);
    const currentSubject = currentSubjectEl ? (currentSubjectEl.innerText || '').trim() : '';

    if (currentSubject && currentSubject !== lastSubject) {
      lastSubject = currentSubject;
      // When user navigates to a new email, clean up previous banner
      removeBanner();
    }
  }, 400);
}

const observer = new MutationObserver(onDomMutated);
observer.observe(document.body, { childList: true, subtree: true });
