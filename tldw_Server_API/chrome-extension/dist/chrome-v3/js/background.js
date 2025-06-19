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
// Extension Update and Installation Handler
class ExtensionUpdateManager {
  constructor() {
    this.currentVersion = chrome.runtime.getManifest().version;
    this.migrationStrategies = new Map();
    this.setupMigrationStrategies();
  }
  
  async handleInstallation(details) {
    const { reason, previousVersion } = details;
    
    switch (reason) {
      case 'install':
        await this.handleFirstInstall();
        break;
      case 'update':
        await this.handleUpdate(previousVersion);
        break;
      case 'chrome_update':
        await this.handleChromeUpdate();
        break;
      case 'shared_module_update':
        await this.handleSharedModuleUpdate();
        break;
      default:
        console.log('Unknown installation reason:', reason);
    }
  }
  
  async handleFirstInstall() {
    console.log('TLDW Extension first install detected');
    
    try {
      // Set default configuration
      await this.setDefaultConfiguration();
      
      // Initialize extension state
      await this.initializeExtensionState();
      
      // Show welcome notification
      this.showWelcomeNotification();
      
      // Open options page for initial setup
      chrome.runtime.openOptionsPage();
      
    } catch (error) {
      console.error('Failed to handle first install:', error);
    }
  }
  
  async handleUpdate(previousVersion) {
    console.log(`TLDW Extension updated from ${previousVersion} to ${this.currentVersion}`);
    
    try {
      // Backup current data before migration
      await this.backupExtensionData();
      
      // Perform version-specific migrations
      await this.migrateData(previousVersion, this.currentVersion);
      
      // Update extension state
      await this.updateExtensionVersion();
      
      // Show update notification
      this.showUpdateNotification(previousVersion);
      
      // Clean up old data if needed
      await this.cleanupOldData();
      
    } catch (error) {
      console.error('Failed to handle update:', error);
      await this.rollbackOnError();
    }
  }
  
  async handleChromeUpdate() {
    console.log('Chrome browser updated');
    
    try {
      // Verify extension compatibility
      await this.verifyCompatibility();
      
      // Refresh connection status
      await this.refreshConnectionStatus();
      
    } catch (error) {
      console.error('Failed to handle Chrome update:', error);
    }
  }
  
  async handleSharedModuleUpdate() {
    console.log('Shared module updated');
    // Handle shared module updates if needed
  }
  
  async setDefaultConfiguration() {
    const defaultConfig = {
      serverUrl: 'http://localhost:8000',
      apiToken: '',
      enableNotifications: true,
      enableFloatingButton: true,
      enableSmartContext: true,
      enableBatchOperations: true,
      enableAdvancedSearch: true,
      enableProgressIndicators: true,
      extensionVersion: this.currentVersion,
      installDate: new Date().toISOString(),
      firstRun: true
    };
    
    // Only set values that don't already exist
    const existing = await chrome.storage.sync.get(Object.keys(defaultConfig));
    const toSet = {};
    
    Object.keys(defaultConfig).forEach(key => {
      if (!(key in existing)) {
        toSet[key] = defaultConfig[key];
      }
    });
    
    if (Object.keys(toSet).length > 0) {
      await chrome.storage.sync.set(toSet);
    }
  }
  
  async initializeExtensionState() {
    // Clear any cached data that might be invalid
    await chrome.storage.local.clear();
    
    // Initialize local state
    await chrome.storage.local.set({
      lastConnectionCheck: null,
      connectionStatus: 'unknown',
      cacheInitialized: false,
      migrationVersion: this.currentVersion
    });
  }
  
  async migrateData(fromVersion, toVersion) {
    console.log(`Migrating data from ${fromVersion} to ${toVersion}`);
    
    const migrations = this.getMigrationsToRun(fromVersion, toVersion);
    
    for (const migration of migrations) {
      try {
        console.log(`Running migration: ${migration.name}`);
        await migration.execute();
        await this.recordMigration(migration);
      } catch (error) {
        console.error(`Migration ${migration.name} failed:`, error);
        throw error;
      }
    }
  }
  
  setupMigrationStrategies() {
    // Example migrations for different version updates
    this.migrationStrategies.set('1.0.0->1.1.0', {
      name: 'Add new configuration options',
      execute: async () => {
        const newOptions = {
          enableProgressIndicators: true,
          maxBatchSize: 50
        };
        await chrome.storage.sync.set(newOptions);
      }
    });
    
    this.migrationStrategies.set('1.1.0->1.2.0', {
      name: 'Migrate API token format',
      execute: async () => {
        const { apiKey } = await chrome.storage.sync.get(['apiKey']);
        if (apiKey) {
          await chrome.storage.sync.set({ apiToken: apiKey });
          await chrome.storage.sync.remove(['apiKey']);
        }
      }
    });
    
    this.migrationStrategies.set('1.2.0->2.0.0', {
      name: 'Major configuration restructure',
      execute: async () => {
        // Migrate to new configuration structure
        const oldConfig = await chrome.storage.sync.get();
        const newConfig = this.transformToNewConfigFormat(oldConfig);
        await chrome.storage.sync.clear();
        await chrome.storage.sync.set(newConfig);
      }
    });
  }
  
  getMigrationsToRun(fromVersion, toVersion) {
    const migrations = [];
    
    // Simple version comparison - in production, use proper semver
    this.migrationStrategies.forEach((migration, versionRange) => {
      const [from, to] = versionRange.split('->');
      if (this.shouldRunMigration(fromVersion, toVersion, from, to)) {
        migrations.push(migration);
      }
    });
    
    return migrations.sort((a, b) => a.name.localeCompare(b.name));
  }
  
  shouldRunMigration(currentFrom, currentTo, migrationFrom, migrationTo) {
    // Simplified version comparison - use semver library in production
    return this.compareVersions(currentFrom, migrationFrom) >= 0 && 
           this.compareVersions(currentTo, migrationTo) >= 0;
  }
  
  compareVersions(a, b) {
    const aParts = a.split('.').map(Number);
    const bParts = b.split('.').map(Number);
    
    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      const aPart = aParts[i] || 0;
      const bPart = bParts[i] || 0;
      
      if (aPart > bPart) return 1;
      if (aPart < bPart) return -1;
    }
    
    return 0;
  }
  
  async backupExtensionData() {
    try {
      const syncData = await chrome.storage.sync.get();
      const localData = await chrome.storage.local.get();
      
      const backup = {
        version: this.currentVersion,
        timestamp: new Date().toISOString(),
        syncData,
        localData
      };
      
      await chrome.storage.local.set({ 
        [`backup_${Date.now()}`]: backup 
      });
      
      // Keep only the last 3 backups
      await this.cleanupOldBackups();
      
    } catch (error) {
      console.error('Failed to backup extension data:', error);
    }
  }
  
  async cleanupOldBackups() {
    const localData = await chrome.storage.local.get();
    const backupKeys = Object.keys(localData).filter(key => key.startsWith('backup_'));
    
    if (backupKeys.length > 3) {
      // Sort by timestamp and remove oldest
      backupKeys.sort().slice(0, -3).forEach(key => {
        chrome.storage.local.remove(key);
      });
    }
  }
  
  async rollbackOnError() {
    try {
      // Find the most recent backup
      const localData = await chrome.storage.local.get();
      const backupKeys = Object.keys(localData).filter(key => key.startsWith('backup_'));
      
      if (backupKeys.length > 0) {
        const latestBackupKey = backupKeys.sort().pop();
        const backup = localData[latestBackupKey];
        
        // Restore from backup
        await chrome.storage.sync.clear();
        await chrome.storage.sync.set(backup.syncData);
        
        console.log('Rollback completed from backup:', latestBackupKey);
      }
    } catch (error) {
      console.error('Rollback failed:', error);
    }
  }
  
  async updateExtensionVersion() {
    await chrome.storage.sync.set({
      extensionVersion: this.currentVersion,
      lastUpdated: new Date().toISOString()
    });
  }
  
  async recordMigration(migration) {
    const migrations = await chrome.storage.local.get(['completedMigrations']);
    const completed = migrations.completedMigrations || [];
    
    completed.push({
      name: migration.name,
      timestamp: new Date().toISOString(),
      version: this.currentVersion
    });
    
    await chrome.storage.local.set({ completedMigrations: completed });
  }
  
  async cleanupOldData() {
    // Remove temporary migration data
    const keysToRemove = ['tempMigrationData', 'migrationInProgress'];
    await chrome.storage.local.remove(keysToRemove);
    
    // Clean up old cache entries
    await this.cleanupCache();
  }
  
  async cleanupCache() {
    const localData = await chrome.storage.local.get();
    const cacheKeys = Object.keys(localData).filter(key => 
      key.startsWith('cache_') || key.startsWith('temp_')
    );
    
    if (cacheKeys.length > 0) {
      await chrome.storage.local.remove(cacheKeys);
    }
  }
  
  async verifyCompatibility() {
    // Check if the current Chrome version is supported
    const chromeVersion = this.getChromeVersion();
    const manifest = chrome.runtime.getManifest();
    
    if (manifest.minimum_chrome_version) {
      const minVersion = manifest.minimum_chrome_version;
      if (this.compareVersions(chromeVersion, minVersion) < 0) {
        console.warn(`Chrome version ${chromeVersion} is below minimum required ${minVersion}`);
      }
    }
  }
  
  getChromeVersion() {
    const userAgent = navigator.userAgent;
    const match = userAgent.match(/Chrome\/([0-9.]+)/);
    return match ? match[1] : 'unknown';
  }
  
  async refreshConnectionStatus() {
    // Reset connection status after browser update
    await chrome.storage.local.set({
      lastConnectionCheck: null,
      connectionStatus: 'unknown'
    });
  }
  
  showWelcomeNotification() {
    chrome.notifications.create('welcome', {
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: 'TLDW Extension Installed',
      message: 'Welcome! Click to configure your server settings.'
    }, () => {
      // Handle notification click
      chrome.notifications.onClicked.addListener((notificationId) => {
        if (notificationId === 'welcome') {
          chrome.runtime.openOptionsPage();
        }
      });
    });
  }
  
  showUpdateNotification(previousVersion) {
    chrome.notifications.create('update', {
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: 'TLDW Extension Updated',
      message: `Updated from v${previousVersion} to v${this.currentVersion}. Check what's new!`
    });
  }
  
  transformToNewConfigFormat(oldConfig) {
    // Example transformation for major version changes
    return {
      ...oldConfig,
      version: '2.0.0',
      migrated: true,
      migratedAt: new Date().toISOString()
    };
  }
}

// Initialize update manager
const updateManager = new ExtensionUpdateManager();

// Enhanced installation listener
chrome.runtime.onInstalled.addListener(async (details) => {
  await updateManager.handleInstallation(details);
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