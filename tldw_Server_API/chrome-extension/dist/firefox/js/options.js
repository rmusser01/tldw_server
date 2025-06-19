// Browser API compatibility
if (typeof browserAPI === 'undefined') {
  window.browserAPI = (typeof browser !== 'undefined') ? browser : chrome;
}

// Options page functionality
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  setupEventListeners();
  checkForPendingPrompt();
});

async function loadSettings() {
  const settings = await browserAPI.storage.sync.get([
    'serverUrl',
    'apiToken',
    'defaultModel',
    'defaultTemperature',
    'maxTokens',
    'autoLoadChats',
    'streamResponses',
    'showNotifications'
  ]);

  // Populate form fields
  document.getElementById('serverUrl').value = settings.serverUrl || 'http://localhost:8000';
  document.getElementById('apiToken').value = settings.apiToken || '';
  document.getElementById('defaultModel').value = settings.defaultModel || '';
  document.getElementById('defaultTemperature').value = settings.defaultTemperature || 0.7;
  document.getElementById('temperatureValue').textContent = settings.defaultTemperature || 0.7;
  document.getElementById('maxTokens').value = settings.maxTokens || 1000;
  document.getElementById('autoLoadChats').checked = settings.autoLoadChats || false;
  document.getElementById('streamResponses').checked = settings.streamResponses || false;
  document.getElementById('showNotifications').checked = settings.showNotifications !== false;
}

function setupEventListeners() {
  // Save settings
  document.getElementById('saveSettings').addEventListener('click', saveSettings);
  document.getElementById('cancelSettings').addEventListener('click', () => window.close());

  // Test connection
  document.getElementById('testConnection').addEventListener('click', testConnection);

  // Temperature slider
  document.getElementById('defaultTemperature').addEventListener('input', (e) => {
    document.getElementById('temperatureValue').textContent = e.target.value;
  });

  // Data management
  document.getElementById('exportSettings').addEventListener('click', exportSettings);
  document.getElementById('importSettings').addEventListener('click', () => {
    document.getElementById('importFile').click();
  });
  document.getElementById('importFile').addEventListener('change', importSettings);
  document.getElementById('clearCache').addEventListener('click', clearCache);
  document.getElementById('resetSettings').addEventListener('click', resetSettings);

  // Prompt creation
  document.getElementById('savePrompt').addEventListener('click', savePrompt);
  document.getElementById('cancelPrompt').addEventListener('click', () => {
    document.getElementById('promptCreation').style.display = 'none';
    chrome.storage.local.remove(['pendingPromptText']);
  });
}

async function saveSettings() {
  const settings = {
    serverUrl: document.getElementById('serverUrl').value,
    apiToken: document.getElementById('apiToken').value,
    defaultModel: document.getElementById('defaultModel').value,
    defaultTemperature: parseFloat(document.getElementById('defaultTemperature').value),
    maxTokens: parseInt(document.getElementById('maxTokens').value),
    autoLoadChats: document.getElementById('autoLoadChats').checked,
    streamResponses: document.getElementById('streamResponses').checked,
    showNotifications: document.getElementById('showNotifications').checked
  };

  try {
    await browserAPI.storage.sync.set(settings);
    showStatus('Settings saved successfully!', 'success');
    
    // Notify background script of settings change
    browserAPI.runtime.sendMessage({ action: 'settingsUpdated' });
    
    setTimeout(() => window.close(), 1500);
  } catch (error) {
    showStatus('Failed to save settings', 'error');
  }
}

async function testConnection() {
  const serverUrl = document.getElementById('serverUrl').value;
  const apiToken = document.getElementById('apiToken').value;

  if (!serverUrl) {
    showStatus('Please enter a server URL', 'error');
    return;
  }

  showStatus('Testing connection...', 'info');

  try {
    const response = await fetch(`${serverUrl}/api/v1/media/`, {
      method: 'GET',
      headers: {
        'Token': `Bearer ${apiToken}`
      }
    });

    if (response.ok) {
      showStatus('Connection successful!', 'success');
    } else if (response.status === 401) {
      showStatus('Authentication failed. Check your API token.', 'error');
    } else {
      showStatus(`Connection failed: HTTP ${response.status}`, 'error');
    }
  } catch (error) {
    showStatus('Connection failed. Check server URL and ensure server is running.', 'error');
  }
}

function showStatus(message, type) {
  const statusElement = document.getElementById('connectionStatus');
  statusElement.textContent = message;
  statusElement.className = `status-message ${type}`;
}

async function exportSettings() {
  const settings = await browserAPI.storage.sync.get(null);
  const dataStr = JSON.stringify(settings, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  
  const url = URL.createObjectURL(dataBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'tldw-assistant-settings.json';
  a.click();
  URL.revokeObjectURL(url);
}

async function importSettings(event) {
  const file = event.target.files[0];
  if (!file) return;

  try {
    const text = await file.text();
    const settings = JSON.parse(text);
    
    await browserAPI.storage.sync.set(settings);
    await loadSettings();
    showStatus('Settings imported successfully!', 'success');
  } catch (error) {
    showStatus('Failed to import settings', 'error');
  }
}

async function clearCache() {
  if (confirm('Are you sure you want to clear all cached data?')) {
    await browserAPI.storage.local.clear();
    showStatus('Cache cleared successfully!', 'success');
  }
}

async function resetSettings() {
  if (confirm('Are you sure you want to reset all settings to defaults?')) {
    await browserAPI.storage.sync.clear();
    await browserAPI.storage.local.clear();
    await loadSettings();
    showStatus('Settings reset to defaults!', 'success');
  }
}

async function checkForPendingPrompt() {
  const { pendingPromptText } = await browserAPI.storage.local.get(['pendingPromptText']);
  
  if (pendingPromptText) {
    document.getElementById('promptCreation').style.display = 'block';
    document.getElementById('promptContent').value = pendingPromptText;
    document.getElementById('promptName').focus();
  }
}

async function savePrompt() {
  const name = document.getElementById('promptName').value.trim();
  const content = document.getElementById('promptContent').value.trim();
  
  if (!name || !content) {
    alert('Please enter both a name and content for the prompt');
    return;
  }

  try {
    const settings = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
    const response = await fetch(`${settings.serverUrl}/api/v1/prompts/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Token': `Bearer ${settings.apiToken}`
      },
      body: JSON.stringify({
        name: name,
        system_prompt: content,
        details: 'Created from browser extension'
      })
    });

    if (response.ok) {
      alert('Prompt saved successfully!');
      document.getElementById('promptCreation').style.display = 'none';
      document.getElementById('promptName').value = '';
      document.getElementById('promptContent').value = '';
      browserAPI.storage.local.remove(['pendingPromptText']);
    } else {
      const error = await response.json();
      alert(`Failed to save prompt: ${error.detail || 'Unknown error'}`);
    }
  } catch (error) {
    alert('Failed to connect to server');
  }
}