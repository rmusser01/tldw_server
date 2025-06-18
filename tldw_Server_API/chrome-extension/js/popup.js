// Browser API compatibility
const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;

// Global state
let currentConversationId = null;
let selectedCharacterId = null;
let selectedPromptTemplate = null;

// Toast notification system
class ToastManager {
  constructor() {
    this.container = null;
    this.toasts = new Set();
  }

  init() {
    this.container = document.getElementById('toast-container');
    if (!this.container) {
      console.error('Toast container not found');
    }
  }

  show(message, type = 'info', title = null, duration = 4000) {
    if (!this.container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = document.createElement('div');
    icon.className = 'toast-icon';
    
    const content = document.createElement('div');
    content.className = 'toast-content';
    
    if (title) {
      const titleElement = document.createElement('div');
      titleElement.className = 'toast-title';
      titleElement.textContent = title;
      content.appendChild(titleElement);
    }
    
    const messageElement = document.createElement('div');
    messageElement.className = 'toast-message';
    messageElement.textContent = message;
    content.appendChild(messageElement);
    
    const closeButton = document.createElement('button');
    closeButton.className = 'toast-close';
    closeButton.innerHTML = '×';
    closeButton.onclick = () => this.hide(toast);
    
    toast.appendChild(icon);
    toast.appendChild(content);
    toast.appendChild(closeButton);
    
    this.container.appendChild(toast);
    this.toasts.add(toast);
    
    // Auto-hide after duration
    if (duration > 0) {
      setTimeout(() => this.hide(toast), duration);
    }
    
    return toast;
  }

  hide(toast) {
    if (!toast || !this.toasts.has(toast)) return;
    
    toast.classList.add('toast-hiding');
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
      this.toasts.delete(toast);
    }, 300);
  }

  success(message, title = 'Success') {
    return this.show(message, 'success', title);
  }

  error(message, title = 'Error') {
    return this.show(message, 'error', title, 6000);
  }

  warning(message, title = 'Warning') {
    return this.show(message, 'warning', title, 5000);
  }

  info(message, title = null) {
    return this.show(message, 'info', title);
  }

  loading(message, title = 'Loading...') {
    const toast = this.show(message, 'info', title, 0);
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    toast.querySelector('.toast-icon').appendChild(spinner);
    return toast;
  }
}

const toast = new ToastManager();

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
  await initializePopup();
  setupEventListeners();
  setupTabs();
});

async function initializePopup() {
  // Initialize toast system
  toast.init();
  
  // Check connection status
  const isConnected = await apiClient.checkConnection();
  const status = apiClient.getConnectionStatus();
  updateConnectionStatus(isConnected, status);
  
  // Start connection monitoring
  startConnectionMonitoring();
  
  // Load initial data
  if (isConnected) {
    loadPrompts();
    loadCharacters();
    loadMediaList();
    toast.success('Connected to TLDW Server', 'Connected');
  } else {
    toast.error('Failed to connect to TLDW Server. Check your settings.', 'Connection Failed');
  }
}

function updateConnectionStatus(isConnected, statusInfo = null) {
  const statusDot = document.getElementById('connectionStatus');
  const statusText = document.getElementById('connectionText');
  
  if (isConnected) {
    statusDot.classList.add('connected');
    statusDot.classList.remove('error', 'warning');
    statusText.textContent = 'Connected';
    statusText.title = statusInfo?.lastSuccessful ? 
      `Last successful: ${statusInfo.lastSuccessful.toLocaleTimeString()}` : 
      'Connected to TLDW Server';
  } else {
    statusDot.classList.add('error');
    statusDot.classList.remove('connected', 'warning');
    
    if (statusInfo?.consecutiveFailures > 0) {
      statusText.textContent = `Disconnected (${statusInfo.consecutiveFailures} failures)`;
      statusText.title = statusInfo.lastError ? 
        `Last error: ${statusInfo.lastError.message}` : 
        'Connection failed';
    } else {
      statusText.textContent = 'Disconnected';
      statusText.title = 'Not connected to TLDW Server';
    }
  }
}

// Enhanced connection monitoring
let connectionCheckInterval = null;
let isRetrying = false;

function startConnectionMonitoring() {
  // Check connection every 30 seconds
  connectionCheckInterval = setInterval(async () => {
    if (isRetrying) return; // Don't overlap with manual retries
    
    const isConnected = await apiClient.checkConnection();
    const status = apiClient.getConnectionStatus();
    updateConnectionStatus(isConnected, status);
  }, 30000);
}

function stopConnectionMonitoring() {
  if (connectionCheckInterval) {
    clearInterval(connectionCheckInterval);
    connectionCheckInterval = null;
  }
}

async function retryConnection() {
  if (isRetrying) return;
  
  isRetrying = true;
  const statusText = document.getElementById('connectionText');
  const originalText = statusText.textContent;
  
  try {
    statusText.textContent = 'Retrying...';
    const isConnected = await apiClient.checkConnection(true); // Enable retry
    const status = apiClient.getConnectionStatus();
    
    updateConnectionStatus(isConnected, status);
    
    if (isConnected) {
      toast.success('Reconnected to TLDW Server');
      // Reload data after reconnection
      loadPrompts();
      loadCharacters();
      loadMediaList();
    } else {
      toast.error('Failed to reconnect. Check your server settings.');
    }
  } catch (error) {
    toast.error(`Connection retry failed: ${error.message}`);
    statusText.textContent = originalText;
  } finally {
    isRetrying = false;
  }
}

// Make updateConnectionStatus available globally for API client callback
window.updateConnectionStatus = updateConnectionStatus;

// Tab functionality
function setupTabs() {
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.dataset.tab;
      
      // Update active states
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabPanes.forEach(pane => pane.classList.remove('active'));
      
      button.classList.add('active');
      document.getElementById(`${targetTab}-tab`).classList.add('active');
    });
  });
}

// Event listeners
function setupEventListeners() {
  // Chat events
  document.getElementById('sendMessage').addEventListener('click', sendChatMessage);
  document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
  document.getElementById('clearChat').addEventListener('click', clearChat);
  document.getElementById('characterSelect').addEventListener('change', (e) => {
    selectedCharacterId = e.target.value || null;
  });
  
  // Prompts events
  document.getElementById('searchPromptsBtn').addEventListener('click', searchPrompts);
  document.getElementById('promptSearch').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchPrompts();
  });
  document.getElementById('createPrompt').addEventListener('click', () => {
    openPromptModal();
  });
  document.getElementById('exportPrompts').addEventListener('click', exportPrompts);
  
  // Characters events
  document.getElementById('searchCharactersBtn').addEventListener('click', searchCharacters);
  document.getElementById('characterSearch').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchCharacters();
  });
  document.getElementById('importCharacter').addEventListener('click', () => {
    document.getElementById('characterFileInput').click();
  });
  document.getElementById('characterFileInput').addEventListener('change', importCharacterFile);
  
  // Media events
  document.getElementById('processUrl').addEventListener('click', processMediaUrl);
  document.getElementById('mediaUrl').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') processMediaUrl();
  });
  document.getElementById('processFile').addEventListener('click', () => {
    document.getElementById('mediaFileInput').click();
  });
  document.getElementById('mediaFileInput').addEventListener('change', processMediaFile);
  
  // Footer events
  document.getElementById('openOptions').addEventListener('click', (e) => {
    e.preventDefault();
    browserAPI.runtime.openOptionsPage();
  });
  document.getElementById('openHelp').addEventListener('click', (e) => {
    e.preventDefault();
    browserAPI.tabs.create({ url: 'https://github.com/rmusser01/tldw_server' });
  });
  
  // Connection status retry
  document.getElementById('connectionText').addEventListener('click', () => {
    const status = apiClient.getConnectionStatus();
    if (!status.isConnected) {
      retryConnection();
    }
  });
  
  // Add cursor pointer style for disconnected status
  document.getElementById('connectionText').style.cursor = 'pointer';
}

// Chat functionality
async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  const model = document.getElementById('modelSelect').value;
  
  if (!message || !model) {
    toast.warning('Please enter a message and select a model');
    return;
  }
  
  // Add user message to chat
  addMessageToChat('user', message);
  input.value = '';
  
  // Prepare messages array
  const messages = [
    { role: 'user', content: message }
  ];
  
  // Prepare request data
  const requestData = {
    model: model,
    messages: messages,
    stream: false
  };
  
  if (selectedCharacterId) {
    requestData.character_id = selectedCharacterId;
  }
  
  if (currentConversationId) {
    requestData.conversation_id = currentConversationId;
  }
  
  if (selectedPromptTemplate) {
    requestData.prompt_template_name = selectedPromptTemplate;
  }
  
  try {
    const response = await apiClient.createChatCompletion(requestData);
    
    if (response.choices && response.choices[0]) {
      const assistantMessage = response.choices[0].message.content;
      addMessageToChat('assistant', assistantMessage);
      
      // Update conversation ID if provided
      if (response.conversation_id) {
        currentConversationId = response.conversation_id;
      }
    }
  } catch (error) {
    console.error('Chat error:', error);
    addMessageToChat('system', `Error: ${error.message}`);
    toast.error(`Chat request failed: ${error.message}`);
  }
}

function addMessageToChat(role, content) {
  const messagesContainer = document.getElementById('chatMessages');
  const template = document.getElementById('message-template').content.cloneNode(true);
  
  const messageDiv = template.querySelector('.message');
  const roleDiv = template.querySelector('.message-role');
  const contentDiv = template.querySelector('.message-content');
  
  roleDiv.textContent = role.charAt(0).toUpperCase() + role.slice(1);
  roleDiv.classList.add(role);
  contentDiv.textContent = content;
  
  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function clearChat() {
  document.getElementById('chatMessages').innerHTML = '';
  currentConversationId = null;
}

// Prompts functionality
async function loadPrompts() {
  try {
    const response = await apiClient.getPrompts(1, 20);
    displayPrompts(response.prompts);
  } catch (error) {
    console.error('Failed to load prompts:', error);
    toast.error('Failed to load prompts from server');
  }
}

async function searchPrompts() {
  const query = document.getElementById('promptSearch').value.trim();
  if (!query) {
    loadPrompts();
    return;
  }
  
  try {
    const response = await apiClient.searchPrompts(query);
    displayPrompts(response.results);
  } catch (error) {
    console.error('Search failed:', error);
    toast.error('Prompt search failed');
  }
}

function displayPrompts(prompts) {
  const container = document.getElementById('promptsList');
  container.innerHTML = '';
  
  if (!prompts || prompts.length === 0) {
    container.innerHTML = '<div class="loading">No prompts found</div>';
    return;
  }
  
  prompts.forEach(prompt => {
    const template = document.getElementById('prompt-item-template').content.cloneNode(true);
    
    template.querySelector('.prompt-name').textContent = prompt.name;
    template.querySelector('.prompt-details').textContent = prompt.details || 'No description';
    
    const keywordsDiv = template.querySelector('.prompt-keywords');
    if (prompt.keywords && prompt.keywords.length > 0) {
      prompt.keywords.forEach(keyword => {
        const tag = document.createElement('span');
        tag.className = 'keyword-tag';
        tag.textContent = keyword;
        keywordsDiv.appendChild(tag);
      });
    }
    
    template.querySelector('.use-prompt').addEventListener('click', () => {
      selectedPromptTemplate = prompt.name;
      toast.success(`Selected prompt: ${prompt.name}`);
    });
    
    template.querySelector('.edit-prompt').addEventListener('click', () => {
      toast.info('Edit functionality is coming soon!');
    });
    
    container.appendChild(template);
  });
}

async function exportPrompts() {
  try {
    const response = await apiClient.exportPrompts('markdown');
    
    // Decode base64 and download
    const content = atob(response.content);
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    
    // Use browserAPI for downloads
    if (browserAPI.downloads) {
      browserAPI.downloads.download({
        url: url,
        filename: response.filename,
        saveAs: true
      });
    } else {
      // Fallback for browsers without downloads API
      const a = document.createElement('a');
      a.href = url;
      a.download = response.filename;
      a.click();
      URL.revokeObjectURL(url);
    }
  } catch (error) {
    console.error('Export failed:', error);
    toast.error('Failed to export prompts');
  }
}

// Characters functionality
async function loadCharacters() {
  try {
    const characters = await apiClient.getCharacters();
    displayCharacters(characters);
    updateCharacterSelect(characters);
  } catch (error) {
    console.error('Failed to load characters:', error);
    toast.error('Failed to load characters from server');
  }
}

async function searchCharacters() {
  const query = document.getElementById('characterSearch').value.trim();
  if (!query) {
    loadCharacters();
    return;
  }
  
  try {
    const characters = await apiClient.searchCharacters(query);
    displayCharacters(characters);
  } catch (error) {
    console.error('Search failed:', error);
    toast.error('Character search failed');
  }
}

function displayCharacters(characters) {
  const container = document.getElementById('charactersList');
  container.innerHTML = '';
  
  if (!characters || characters.length === 0) {
    container.innerHTML = '<div class="loading">No characters found</div>';
    return;
  }
  
  characters.forEach(character => {
    const template = document.getElementById('character-card-template').content.cloneNode(true);
    
    const img = template.querySelector('.character-image');
    if (character.image_base64) {
      img.src = `data:image/png;base64,${character.image_base64}`;
    } else {
      img.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24"%3E%3Cpath fill="%23bdc3c7" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/%3E%3C/svg%3E';
    }
    
    template.querySelector('.character-name').textContent = character.name;
    template.querySelector('.character-description').textContent = character.description || 'No description';
    
    template.querySelector('.select-character').addEventListener('click', () => {
      selectedCharacterId = character.id.toString();
      document.getElementById('characterSelect').value = selectedCharacterId;
      toast.success(`Selected character: ${character.name}`);
    });
    
    container.appendChild(template);
  });
}

function updateCharacterSelect(characters) {
  const select = document.getElementById('characterSelect');
  
  // Clear existing options except the first
  while (select.options.length > 1) {
    select.remove(1);
  }
  
  characters.forEach(character => {
    const option = document.createElement('option');
    option.value = character.id;
    option.textContent = character.name;
    select.appendChild(option);
  });
}

async function importCharacterFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  try {
    const response = await apiClient.importCharacterCard(file);
    toast.success(`Character imported successfully: ${response.character.name}`);
    loadCharacters();
  } catch (error) {
    console.error('Import failed:', error);
    toast.error(`Failed to import character: ${error.message}`);
  }
}

// Media functionality
async function loadMediaList() {
  try {
    const response = await apiClient.getMediaList();
    displayMediaItems(response.media_items);
  } catch (error) {
    console.error('Failed to load media:', error);
    toast.error('Failed to load media items from server');
  }
}

function displayMediaItems(items) {
  const container = document.getElementById('mediaList');
  container.innerHTML = '';
  
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="loading">No media items found</div>';
    return;
  }
  
  items.forEach(item => {
    const template = document.getElementById('media-item-template').content.cloneNode(true);
    
    template.querySelector('.media-title').textContent = item.title;
    template.querySelector('.media-type').textContent = item.type || 'Unknown';
    
    template.querySelector('.view-media').addEventListener('click', () => {
      viewMediaItem(item.id);
    });
    
    template.querySelector('.summarize-media').addEventListener('click', () => {
      summarizeMediaItem(item.id);
    });
    
    container.appendChild(template);
  });
}

async function processMediaUrl() {
  const url = document.getElementById('mediaUrl').value.trim();
  if (!url) {
    toast.warning('Please enter a URL');
    return;
  }
  
  try {
    const response = await apiClient.processMediaUrl(url);
    toast.success('Media processed successfully!');
    loadMediaList();
  } catch (error) {
    console.error('Processing failed:', error);
    toast.error(`Failed to process media: ${error.message}`);
  }
}

async function processMediaFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  let endpoint;
  const fileType = file.type;
  
  if (fileType.startsWith('video/')) {
    endpoint = 'process-videos';
  } else if (fileType.startsWith('audio/')) {
    endpoint = 'process-audios';
  } else if (fileType === 'application/pdf') {
    endpoint = 'process-pdfs';
  } else if (fileType === 'application/epub+zip') {
    endpoint = 'process-ebooks';
  } else {
    endpoint = 'process-documents';
  }
  
  try {
    const response = await apiClient.processMediaFile(file, endpoint);
    toast.success('File processed successfully!');
    loadMediaList();
  } catch (error) {
    console.error('Processing failed:', error);
    toast.error(`Failed to process file: ${error.message}`);
  }
}

async function viewMediaItem(id) {
  try {
    const item = await apiClient.getMediaItem(id);
    // Open in new tab or display in modal
    console.log('Media item:', item);
    toast.info(`Viewing: ${item.title}`);
  } catch (error) {
    console.error('Failed to load media item:', error);
  }
}

async function summarizeMediaItem(id) {
  try {
    // Use the media content with chat API
    const item = await apiClient.getMediaItem(id);
    
    // Switch to chat tab and populate with summarization request
    document.querySelector('[data-tab="chat"]').click();
    document.getElementById('chatInput').value = `Please summarize the following content:\n\n${item.content || item.transcript || 'Content not available'}`;
  } catch (error) {
    console.error('Failed to summarize:', error);
    toast.error('Failed to load content for summarization');
  }
}

// Modal functionality
function openPromptModal() {
  const modal = document.getElementById('prompt-modal');
  modal.style.display = 'flex';
  
  // Reset form
  document.getElementById('prompt-form').reset();
  
  // Setup event listeners
  setupModalEventListeners();
}

function closePromptModal() {
  const modal = document.getElementById('prompt-modal');
  modal.style.display = 'none';
}

function setupModalEventListeners() {
  // Close button
  document.querySelector('.modal-close').onclick = closePromptModal;
  
  // Cancel button
  document.getElementById('cancel-prompt').onclick = closePromptModal;
  
  // Save button
  document.getElementById('save-prompt').onclick = saveNewPrompt;
  
  // Click outside to close
  document.getElementById('prompt-modal').onclick = (e) => {
    if (e.target.id === 'prompt-modal') {
      closePromptModal();
    }
  };
  
  // Escape key to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closePromptModal();
    }
  });
}

async function saveNewPrompt() {
  const name = document.getElementById('prompt-name').value.trim();
  const details = document.getElementById('prompt-details').value.trim();
  const content = document.getElementById('prompt-content').value.trim();
  const keywordsText = document.getElementById('prompt-keywords').value.trim();
  
  if (!name || !content) {
    toast.warning('Name and content are required');
    return;
  }
  
  const keywords = keywordsText ? keywordsText.split(',').map(k => k.trim()).filter(k => k) : [];
  
  const promptData = {
    name: name,
    details: details || null,
    content: content,
    keywords: keywords
  };
  
  try {
    const loadingToast = toast.loading('Creating prompt...');
    const response = await apiClient.createPrompt(promptData);
    toast.hide(loadingToast);
    
    toast.success(`Prompt "${name}" created successfully`);
    closePromptModal();
    loadPrompts(); // Refresh the prompts list
  } catch (error) {
    console.error('Failed to create prompt:', error);
    toast.error(`Failed to create prompt: ${error.message}`);
  }
}