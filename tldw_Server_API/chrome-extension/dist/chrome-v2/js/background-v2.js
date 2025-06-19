// Background script for TLDW Server Assistant (Manifest V2 compatible)
// Works with both Chrome and Firefox using browser polyfill

// Use browser API (polyfilled for Chrome)
const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;

// Create context menu items when extension is installed
browserAPI.runtime.onInstalled.addListener(() => {
  browserAPI.contextMenus.create({
    id: 'send-to-chat',
    title: 'Send to TLDW Chat',
    contexts: ['selection']
  });
  
  browserAPI.contextMenus.create({
    id: 'process-as-media',
    title: 'Process as Media',
    contexts: ['selection', 'link', 'image', 'video', 'audio']
  });
  
  browserAPI.contextMenus.create({
    id: 'save-as-prompt',
    title: 'Save as Prompt',
    contexts: ['selection']
  });
});

// Handle context menu clicks
browserAPI.contextMenus.onClicked.addListener(async (info, tab) => {
  switch (info.menuItemId) {
    case 'send-to-chat':
      await handleSendToChat(info.selectionText, tab);
      break;
    case 'process-as-media':
      await handleProcessAsMedia(info, tab);
      break;
    case 'save-as-prompt':
      await handleSaveAsPrompt(info.selectionText, tab);
      break;
  }
});

// Handle keyboard shortcuts
browserAPI.commands.onCommand.addListener(async (command) => {
  if (command === 'send-to-chat') {
    const tabs = await browserAPI.tabs.query({ active: true, currentWindow: true });
    const tab = tabs[0];
    
    browserAPI.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
      if (response && response.text) {
        await handleSendToChat(response.text, tab);
      }
    });
  }
});

// Message handler for communication with content scripts and popup
browserAPI.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'processSelection') {
    handleProcessSelection(request.data, sender.tab);
    sendResponse({ success: true });
  } else if (request.action === 'apiRequest') {
    // Proxy API requests from content scripts
    handleApiRequest(request.endpoint, request.options)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep message channel open for async response
  } else if (request.action === 'openPopup') {
    // For Firefox, we need to use browserAction.openPopup
    if (browserAPI.browserAction && browserAPI.browserAction.openPopup) {
      browserAPI.browserAction.openPopup();
    }
    sendResponse({ success: true });
  }
  return false;
});

// Helper functions
async function handleSendToChat(text, tab) {
  if (!text) return;
  
  // Store the selected text temporarily
  await browserAPI.storage.local.set({
    pendingChatText: text,
    sourceUrl: tab.url,
    sourceTitle: tab.title
  });
  
  // Open the popup - different approach for Firefox vs Chrome
  if (browserAPI.browserAction && browserAPI.browserAction.openPopup) {
    // Firefox supports programmatic popup opening
    browserAPI.browserAction.openPopup();
  } else {
    // Chrome doesn't support programmatic popup opening in V2
    // Show notification to user
    showNotification('Text Ready', 'Click the extension icon to send text to chat');
  }
}

async function handleProcessAsMedia(info, tab) {
  let url;
  let type = 'text';
  
  if (info.linkUrl) {
    url = info.linkUrl;
    type = 'link';
  } else if (info.srcUrl) {
    url = info.srcUrl;
    type = info.mediaType || 'media';
  } else if (info.pageUrl) {
    url = info.pageUrl;
    type = 'page';
  }
  
  if (url) {
    try {
      const config = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
      const response = await fetch(`${config.serverUrl || 'http://localhost:8000'}/api/v1/media/ingest-web-content`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Token': `Bearer ${config.apiToken || ''}`
        },
        body: JSON.stringify({
          url: url,
          title: tab.title,
          media_type: type,
          selected_text: info.selectionText
        })
      });
      
      if (response.ok) {
        showNotification('Success', 'Content processed successfully');
      } else {
        showNotification('Error', 'Failed to process content');
      }
    } catch (error) {
      console.error('Processing error:', error);
      showNotification('Error', 'Failed to connect to server');
    }
  }
}

async function handleSaveAsPrompt(text, tab) {
  if (!text) return;
  
  // Store the selected text for prompt creation
  await browserAPI.storage.local.set({
    pendingPromptText: text,
    sourceUrl: tab.url,
    sourceTitle: tab.title
  });
  
  // Open options page
  browserAPI.runtime.openOptionsPage();
}

async function handleApiRequest(endpoint, options) {
  const config = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
  const baseUrl = config.serverUrl || 'http://localhost:8000';
  const apiToken = config.apiToken || '';
  
  const response = await fetch(`${baseUrl}/api/v1${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Token': `Bearer ${apiToken}`,
      ...options.headers
    }
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  
  return response.json();
}

function handleProcessSelection(data, tab) {
  switch (data.type) {
    case 'sendToChat':
      handleSendToChat(data.text, tab);
      break;
    case 'processAsMedia':
      handleProcessAsMedia({
        selectionText: data.text,
        pageUrl: data.url
      }, tab);
      break;
    case 'saveAsPrompt':
      handleSaveAsPrompt(data.text, tab);
      break;
  }
}

function showNotification(title, message) {
  // Use browser notifications API (works in both Chrome and Firefox)
  if (browserAPI.notifications) {
    browserAPI.notifications.create({
      type: 'basic',
      iconUrl: '../icons/icon-48.png',
      title: title,
      message: message
    });
  }
}

// Handle extension installation and updates
browserAPI.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    // Open options page on first install
    browserAPI.runtime.openOptionsPage();
  }
});

// Periodic connection check
setInterval(async () => {
  try {
    const config = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
    const response = await fetch(`${config.serverUrl || 'http://localhost:8000'}/api/v1/media/`, {
      method: 'GET',
      headers: {
        'Token': `Bearer ${config.apiToken || ''}`
      }
    });
    
    // Update badge based on connection status
    if (response.ok) {
      browserAPI.browserAction.setBadgeText({ text: '' });
    } else {
      browserAPI.browserAction.setBadgeText({ text: '!' });
      browserAPI.browserAction.setBadgeBackgroundColor({ color: '#e74c3c' });
    }
  } catch (error) {
    browserAPI.browserAction.setBadgeText({ text: '!' });
    browserAPI.browserAction.setBadgeBackgroundColor({ color: '#e74c3c' });
  }
}, 30000); // Check every 30 seconds