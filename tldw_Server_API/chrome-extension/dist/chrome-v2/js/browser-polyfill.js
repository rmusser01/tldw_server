// Browser API Polyfill
// This provides a unified API that works in both Chrome and Firefox
// It wraps Chrome's callback-based APIs with Promises for Firefox compatibility

(function() {
  'use strict';

  // If browser API already exists (Firefox), we're done
  if (typeof browser !== 'undefined' && browser.runtime) {
    return;
  }

  // Create browser API wrapper for Chrome
  const _browserAPI = {
    runtime: {
      onInstalled: chrome.runtime.onInstalled,
      onMessage: chrome.runtime.onMessage,
      sendMessage: promisify(chrome.runtime.sendMessage),
      openOptionsPage: promisify(chrome.runtime.openOptionsPage),
      getURL: chrome.runtime.getURL,
      id: chrome.runtime.id
    },
    
    storage: {
      sync: {
        get: promisify(chrome.storage.sync.get),
        set: promisify(chrome.storage.sync.set),
        clear: promisify(chrome.storage.sync.clear),
        remove: promisify(chrome.storage.sync.remove)
      },
      local: {
        get: promisify(chrome.storage.local.get),
        set: promisify(chrome.storage.local.set),
        clear: promisify(chrome.storage.local.clear),
        remove: promisify(chrome.storage.local.remove)
      }
    },
    
    tabs: {
      query: promisify(chrome.tabs.query),
      sendMessage: promisify(chrome.tabs.sendMessage),
      create: promisify(chrome.tabs.create),
      get: promisify(chrome.tabs.get),
      update: promisify(chrome.tabs.update),
      remove: promisify(chrome.tabs.remove)
    },
    
    contextMenus: {
      create: chrome.contextMenus.create,
      update: promisify(chrome.contextMenus.update),
      remove: promisify(chrome.contextMenus.remove),
      removeAll: promisify(chrome.contextMenus.removeAll),
      onClicked: chrome.contextMenus.onClicked
    },
    
    // Manifest V2 uses browserAction, V3 uses action
    browserAction: chrome.browserAction ? {
      setBadgeText: promisify(chrome.browserAction.setBadgeText),
      setBadgeBackgroundColor: promisify(chrome.browserAction.setBadgeBackgroundColor),
      setIcon: promisify(chrome.browserAction.setIcon),
      setTitle: promisify(chrome.browserAction.setTitle),
      openPopup: chrome.browserAction.openPopup ? promisify(chrome.browserAction.openPopup) : undefined
    } : undefined,
    
    action: chrome.action ? {
      setBadgeText: promisify(chrome.action.setBadgeText),
      setBadgeBackgroundColor: promisify(chrome.action.setBadgeBackgroundColor),
      setIcon: promisify(chrome.action.setIcon),
      setTitle: promisify(chrome.action.setTitle),
      openPopup: chrome.action.openPopup ? promisify(chrome.action.openPopup) : undefined
    } : undefined,
    
    commands: {
      onCommand: chrome.commands.onCommand
    },
    
    notifications: {
      create: promisify(chrome.notifications.create),
      clear: promisify(chrome.notifications.clear),
      update: promisify(chrome.notifications.update)
    },
    
    downloads: {
      download: promisify(chrome.downloads.download)
    }
  };

  // Helper function to convert callback-based APIs to Promises
  function promisify(fn) {
    if (!fn) return undefined;
    
    return function(...args) {
      return new Promise((resolve, reject) => {
        fn.call(chrome, ...args, function(result) {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(result);
          }
        });
      });
    };
  }

  // Expose browser API globally
  window.browser = _browserAPI;
})();

// Additional compatibility helpers
const isFirefox = navigator.userAgent.includes('Firefox');
const isChrome = navigator.userAgent.includes('Chrome');

// Helper to get the appropriate action API (browserAction for V2, action for V3)
function getActionAPI() {
  if (typeof browser !== 'undefined') {
    return browser.browserAction || browser.action;
  } else if (typeof chrome !== 'undefined') {
    return chrome.browserAction || chrome.action;
  }
  return null;
}

// Export helpers for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { isFirefox, isChrome, getActionAPI };
}