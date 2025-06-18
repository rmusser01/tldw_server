// Background service worker for TLDW Server Assistant

// Create context menu items when extension is installed
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'send-to-chat',
    title: 'Send to TLDW Chat',
    contexts: ['selection']
  });
  
  chrome.contextMenus.create({
    id: 'process-as-media',
    title: 'Process as Media',
    contexts: ['selection', 'link', 'image', 'video', 'audio']
  });
  
  chrome.contextMenus.create({
    id: 'save-as-prompt',
    title: 'Save as Prompt',
    contexts: ['selection']
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
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
chrome.commands.onCommand.addListener(async (command) => {
  if (command === 'send-to-chat') {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
      if (response && response.text) {
        await handleSendToChat(response.text, tab);
      }
    });
  }
});

// Message handler for communication with content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'processSelection') {
    handleProcessSelection(request.data, sender.tab);
    sendResponse({ success: true });
  } else if (request.action === 'apiRequest') {
    // Proxy API requests from content scripts
    handleApiRequest(request.endpoint, request.options)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep message channel open for async response
  }
});

// Helper functions
async function handleSendToChat(text, tab) {
  if (!text) return;
  
  // Store the selected text temporarily
  await chrome.storage.local.set({
    pendingChatText: text,
    sourceUrl: tab.url,
    sourceTitle: tab.title
  });
  
  // Open the popup or focus existing popup
  chrome.action.openPopup();
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
      const config = await chrome.storage.sync.get(['serverUrl', 'apiToken']);
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
  await chrome.storage.local.set({
    pendingPromptText: text,
    sourceUrl: tab.url,
    sourceTitle: tab.title
  });
  
  // Open options page to prompt creation section
  chrome.runtime.openOptionsPage();
}

async function handleApiRequest(endpoint, options) {
  const config = await chrome.storage.sync.get(['serverUrl', 'apiToken']);
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

function showNotification(title, message) {
  chrome.notifications.create({
    type: 'basic',
    iconUrl: '../icons/icon-48.png',
    title: title,
    message: message
  });
}

// Handle extension installation and updates
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    // Open options page on first install
    chrome.runtime.openOptionsPage();
  }
});

// Periodic connection check
setInterval(async () => {
  try {
    const config = await chrome.storage.sync.get(['serverUrl', 'apiToken']);
    const response = await fetch(`${config.serverUrl || 'http://localhost:8000'}/api/v1/media/`, {
      method: 'GET',
      headers: {
        'Token': `Bearer ${config.apiToken || ''}`
      }
    });
    
    // Update badge based on connection status
    if (response.ok) {
      chrome.action.setBadgeText({ text: '' });
    } else {
      chrome.action.setBadgeText({ text: '!' });
      chrome.action.setBadgeBackgroundColor({ color: '#e74c3c' });
    }
  } catch (error) {
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#e74c3c' });
  }
}, 30000); // Check every 30 seconds