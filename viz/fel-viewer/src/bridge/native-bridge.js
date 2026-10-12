// Bidirectional Host Bridge for macOS Cocoa (WKWebView) and Browser CustomEvents

export class NativeBridge {
  constructor(hostElement) {
    this.host = hostElement;
  }

  notifySiteSelected(siteResult, substitutionEvents) {
    const payload = {
      action: 'siteSelected',
      site: siteResult.site,
      alpha: siteResult.alpha,
      beta: siteResult.beta,
      beta_bg: siteResult.beta_bg,
      omega: siteResult.omega,
      pValue: siteResult.pValue,
      lrt: siteResult.lrt,
      classification: siteResult.classification,
      meanSurprise: siteResult.meanSurprise,
      events: substitutionEvents || []
    };

    // 1. Dispatch WebKit Message to macOS Cocoa
    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.hyphyBridge) {
      try {
        window.webkit.messageHandlers.hyphyBridge.postMessage(payload);
      } catch (err) {
        console.warn('[HyPhyBridge] Failed to post message to WKWebView:', err);
      }
    }

    // 2. Dispatch W3C CustomEvent on the Host Element
    if (this.host) {
      this.host.dispatchEvent(new CustomEvent('hyphy:site-select', {
        detail: payload,
        bubbles: true,
        composed: true
      }));
    }
  }

  notifyBranchSelected(branchName, siteEvents) {
    const payload = {
      action: 'branchSelected',
      branch: branchName,
      events: siteEvents || []
    };

    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.hyphyBridge) {
      try {
        window.webkit.messageHandlers.hyphyBridge.postMessage(payload);
      } catch (err) {
        console.warn('[HyPhyBridge] Failed to post message to WKWebView:', err);
      }
    }

    if (this.host) {
      this.host.dispatchEvent(new CustomEvent('hyphy:branch-select', {
        detail: payload,
        bubbles: true,
        composed: true
      }));
    }
  }

  notifyThemeChanged(theme) {
    const payload = {
      action: 'themeChanged',
      theme
    };

    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.hyphyBridge) {
      window.webkit.messageHandlers.hyphyBridge.postMessage(payload);
    }
  }
}
