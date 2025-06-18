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
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  switch (command) {
    case 'send-to-chat':
      chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
        if (response && response.text) {
          await handleSendToChat(response.text, tab);
        } else {
          showNotification('No Text Selected', 'Please select some text first');
        }
      });
      break;
      
    case 'save-as-prompt':
      chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
        if (response && response.text) {
          await handleSaveAsPrompt(response.text, tab);
        } else {
          showNotification('No Text Selected', 'Please select some text first');
        }
      });
      break;
      
    case 'process-page':
      await handleProcessAsMedia({ pageUrl: tab.url }, tab);
      break;
      
    case 'quick-summarize':
      chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (response) => {
        if (response && response.text) {
          await handleQuickSummarize(response.text, tab);
        } else {
          showNotification('No Text Selected', 'Please select some text first');
        }
      });
      break;
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
  
  // Open popup to prompt creation
  chrome.action.openPopup();
  showNotification('Text Saved', 'Selected text ready for prompt creation');
}

async function handleQuickSummarize(text, tab) {
  if (!text) return;
  
  try {
    const config = await chrome.storage.sync.get(['serverUrl', 'apiToken']);
    const response = await fetch(`${config.serverUrl || 'http://localhost:8000'}/api/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Token': `Bearer ${config.apiToken || ''}`
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          {
            role: 'user',
            content: `Please provide a concise summary of the following text:\n\n${text}`
          }
        ],
        stream: false
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      const summary = data.choices?.[0]?.message?.content || 'No summary generated';
      
      // Store summary for popup display
      await chrome.storage.local.set({
        quickSummary: summary,
        summaryText: text,
        sourceUrl: tab.url,
        sourceTitle: tab.title
      });
      
      showNotification('Summary Ready', 'Quick summary generated successfully');
      chrome.action.openPopup();
    } else {
      showNotification('Summary Failed', 'Failed to generate summary');
    }
  } catch (error) {
    console.error('Quick summarize error:', error);
    showNotification('Summary Error', 'Failed to connect to server');
  }
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

// Enhanced periodic connection check with backoff
let connectionFailures = 0;
let maxFailures = 5;

function scheduleConnectionCheck(delay = 30000) {
  setTimeout(async () => {
    try {
      const config = await chrome.storage.sync.get(['serverUrl', 'apiToken']);
      const response = await fetch(`${config.serverUrl || 'http://localhost:8000'}/api/v1/media/`, {
        method: 'GET',
        headers: {
          'Token': `Bearer ${config.apiToken || ''}`
        },
        timeout: 10000
      });
      
      // Update badge based on connection status
      if (response.ok) {
        chrome.action.setBadgeText({ text: '' });
        connectionFailures = 0;
        scheduleConnectionCheck(30000); // Normal interval
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      connectionFailures++;
      chrome.action.setBadgeText({ text: '!' });
      chrome.action.setBadgeBackgroundColor({ color: '#e74c3c' });
      
      // Exponential backoff for failed connections
      const nextDelay = connectionFailures < maxFailures ? 
        Math.min(30000 * Math.pow(2, connectionFailures - 1), 300000) : // Max 5 minutes
        300000; // Check every 5 minutes after max failures
      
      console.log(`Connection check failed (${connectionFailures}/${maxFailures}), next check in ${nextDelay/1000}s`);
      scheduleConnectionCheck(nextDelay);
    }
  }, delay);
}

// Start connection monitoring
scheduleConnectionCheck(5000); // Initial check after 5 seconds