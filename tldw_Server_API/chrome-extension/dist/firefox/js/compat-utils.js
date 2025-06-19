// Compatibility utilities for cross-browser extension development

// Get the appropriate browser API
function getBrowserAPI() {
  return (typeof browser !== 'undefined') ? browser : chrome;
}

// Get the appropriate action API (browserAction for V2, action for V3)
function getActionAPI() {
  const browserAPI = getBrowserAPI();
  return browserAPI.browserAction || browserAPI.action;
}

// Check browser type
const isFirefox = navigator.userAgent.includes('Firefox');
const isChrome = navigator.userAgent.includes('Chrome');
const isEdge = navigator.userAgent.includes('Edg');

// Check manifest version
function getManifestVersion() {
  const browserAPI = getBrowserAPI();
  return browserAPI.runtime.getManifest().manifest_version;
}

// Storage wrapper with error handling
const storage = {
  async get(keys, storageArea = 'sync') {
    const browserAPI = getBrowserAPI();
    try {
      return await browserAPI.storage[storageArea].get(keys);
    } catch (error) {
      console.error(`Storage get error (${storageArea}):`, error);
      return {};
    }
  },

  async set(data, storageArea = 'sync') {
    const browserAPI = getBrowserAPI();
    try {
      await browserAPI.storage[storageArea].set(data);
      return true;
    } catch (error) {
      console.error(`Storage set error (${storageArea}):`, error);
      return false;
    }
  },

  async remove(keys, storageArea = 'sync') {
    const browserAPI = getBrowserAPI();
    try {
      await browserAPI.storage[storageArea].remove(keys);
      return true;
    } catch (error) {
      console.error(`Storage remove error (${storageArea}):`, error);
      return false;
    }
  },

  async clear(storageArea = 'sync') {
    const browserAPI = getBrowserAPI();
    try {
      await browserAPI.storage[storageArea].clear();
      return true;
    } catch (error) {
      console.error(`Storage clear error (${storageArea}):`, error);
      return false;
    }
  }
};

// Tabs wrapper
const tabs = {
  async query(queryInfo) {
    const browserAPI = getBrowserAPI();
    return await browserAPI.tabs.query(queryInfo);
  },

  async sendMessage(tabId, message) {
    const browserAPI = getBrowserAPI();
    try {
      return await browserAPI.tabs.sendMessage(tabId, message);
    } catch (error) {
      console.error('Tab message error:', error);
      return null;
    }
  },

  async create(createProperties) {
    const browserAPI = getBrowserAPI();
    return await browserAPI.tabs.create(createProperties);
  }
};

// Runtime wrapper
const runtime = {
  sendMessage(message) {
    const browserAPI = getBrowserAPI();
    return browserAPI.runtime.sendMessage(message);
  },

  onMessage: {
    addListener(callback) {
      const browserAPI = getBrowserAPI();
      browserAPI.runtime.onMessage.addListener(callback);
    }
  },

  openOptionsPage() {
    const browserAPI = getBrowserAPI();
    return browserAPI.runtime.openOptionsPage();
  },

  getURL(path) {
    const browserAPI = getBrowserAPI();
    return browserAPI.runtime.getURL(path);
  }
};

// Downloads wrapper (Firefox requires additional permission)
const downloads = {
  async download(options) {
    const browserAPI = getBrowserAPI();
    if (!browserAPI.downloads) {
      console.error('Downloads API not available');
      return null;
    }
    
    try {
      return await browserAPI.downloads.download(options);
    } catch (error) {
      console.error('Download error:', error);
      // Fallback to creating a link
      const a = document.createElement('a');
      a.href = options.url;
      a.download = options.filename || 'download';
      a.click();
      return null;
    }
  }
};

// Notifications wrapper
const notifications = {
  async create(notificationId, options) {
    const browserAPI = getBrowserAPI();
    if (!browserAPI.notifications) {
      console.log('Notifications API not available');
      return null;
    }
    
    // Firefox requires iconUrl to be a full path
    if (isFirefox && options.iconUrl && !options.iconUrl.startsWith('moz-extension://')) {
      options.iconUrl = browserAPI.runtime.getURL(options.iconUrl);
    }
    
    try {
      return await browserAPI.notifications.create(notificationId || '', options);
    } catch (error) {
      console.error('Notification error:', error);
      return null;
    }
  }
};

// Export utilities
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    getBrowserAPI,
    getActionAPI,
    isFirefox,
    isChrome,
    isEdge,
    getManifestVersion,
    storage,
    tabs,
    runtime,
    downloads,
    notifications
  };
}