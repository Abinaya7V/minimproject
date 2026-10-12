# AI Email Phishing Detector - Chrome Extension (Manifest V3)

Real-time browser extension for detecting email phishing attacks on **Gmail** and **Outlook Web**, backed by a local Flask Machine Learning API.

---

## 1. Folder Structure

```
D:\mp\extension\
├── manifest.json       # Manifest V3 configuration and host permissions
├── background.js       # Background service worker (API caller with timeout & trimming)
├── content.js          # Injected DOM script (extracts subject, sender, body, URLs, injects banner)
├── popup.html          # Extension action popup markup
├── popup.css           # Clean, responsive styles with confidence meters and badges
├── popup.js            # Popup controller logic (health check, state management)
├── README.md           # This guide and debugging instructions
└── icons/              # Extension icons (16px, 48px, 128px)
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

---

## 2. How to Start the Backend Server

Before using the extension, the local Flask prediction server must be running:

1. Open a terminal in VS Code:
   ```powershell
   cd D:\mp
   ```
2. Activate your virtual environment:
   ```powershell
   .\venv\Scripts\activate
   ```
3. Run the Flask server:
   ```powershell
   python app.py
   ```
4. Confirm the server is active by opening `http://127.0.0.1:5000/api/health` in your browser. You should see:
   ```json
   {"status": "ok"}
   ```

---

## 3. How to Load the Extension in Google Chrome

1. Open Google Chrome.
2. Navigate to: `chrome://extensions/`
3. In the top-right corner, turn on **Developer mode**.
4. Click the **Load unpacked** button in the top-left.
5. In the file picker, select the folder:
   ```
   D:\mp\extension
   ```
6. The **AI Email Phishing Detector** card will appear.
7. *(Recommended)* Click the puzzle piece (Extensions) icon on Chrome's toolbar and pin **AI Phishing Detector** for quick access.

---

## 4. Debugging & Troubleshooting Checklist

| Issue | Likely Cause | Fix |
|---|---|---|
| **"Could not establish connection. Receiving end does not exist"** | The Gmail tab was opened *before* loading or reloading the extension. | **Reload the Gmail/Outlook tab (`F5` or `Ctrl+R`)** so `content.js` gets injected into the page. |
| **"Backend: Offline" / Cannot connect** | Flask app is not running at `http://127.0.0.1:5000`. | Run `python app.py` in your terminal with the virtual environment activated. |
| **"You are not currently on an email page"** | The active tab URL is not `mail.google.com` or `outlook.office.com` / `outlook.live.com`. | Switch to an open Gmail or Outlook tab before clicking the extension icon. |
| **"No email message appears to be currently selected"** | You are on the inbox list, not inside an opened message view. | Click on a specific email to open its full conversation view, then open the extension popup. |
| **Service worker inactive** | Chrome MV3 suspends service workers when idle. | Opening the popup or clicking "Analyze" automatically wakes up `background.js`. To inspect logs, go to `chrome://extensions` and click `Inspect views: service worker`. |
| **CORS or Network Error** | Host permissions missing or CORS blocked. | The extension's `manifest.json` specifies `http://127.0.0.1:5000/*` under `host_permissions`, and `app.py` already permits `chrome-extension://*` via `flask-cors`. |

---

## 5. The Part Most Likely to Break (Gmail DOM Selectors) & How to Fix It

### Why it happens:
Google periodically updates the obfuscated class names in Gmail's single-page web app. For example, message body classes or subject headers may evolve over time.

### How to fix it in 2 minutes:
All selectors are deliberately centralized in a single `CONFIG` dictionary at the top of **`D:\mp\extension\content.js`**:

```javascript
const CONFIG = {
  gmail: {
    subjectSelectors: ['h2.hP', 'h2[data-thread-perm-id]', '[role="main"] h2'],
    senderSelectors: ['span.gD', 'span[email]', '[data-hovercard-id]'],
    bodySelectors: ['div.a3s.aiL', 'div.a3s', '.adn.ads div.a3s'],
    bannerContainerSelectors: ['.ha', 'h2.hP']
  },
  ...
};
```

If Gmail changes its layout:
1. Open an email in Gmail.
2. Right-click the subject or body and select **Inspect**.
3. Identify the updated class or attribute.
4. Add the new selector to `subjectSelectors`, `senderSelectors`, or `bodySelectors` in `content.js`.
5. Go to `chrome://extensions` and click the **Reload** button on the extension card.
