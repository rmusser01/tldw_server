// Browser API compatibility
const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;

// Global state
let currentConversationId = null;
let selectedCharacterId = null;
let selectedPromptTemplate = null;

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
  await initializePopup();
  setupEventListeners();
  setupTabs();
});

async function initializePopup() {
  // Check connection status
  const isConnected = await apiClient.checkConnection();
  updateConnectionStatus(isConnected);
  
  // Load initial data
  if (isConnected) {
    loadPrompts();
    loadCharacters();
    loadMediaList();
  }
}

function updateConnectionStatus(isConnected) {
  const statusDot = document.getElementById('connectionStatus');
  const statusText = document.getElementById('connectionText');
  
  if (isConnected) {
    statusDot.classList.add('connected');
    statusDot.classList.remove('error');
    statusText.textContent = 'Connected';
  } else {
    statusDot.classList.add('error');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Disconnected';
  }
}

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
    // Open prompt creation dialog
    alert('Prompt creation UI coming soon!');
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
}

// Chat functionality
async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  const model = document.getElementById('modelSelect').value;
  
  if (!message || !model) {
    alert('Please enter a message and select a model');
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
      alert(`Selected prompt: ${prompt.name}`);
    });
    
    template.querySelector('.edit-prompt').addEventListener('click', () => {
      alert('Edit functionality coming soon!');
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
    alert('Failed to export prompts');
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
      alert(`Selected character: ${character.name}`);
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
    alert(`Character imported successfully: ${response.character.name}`);
    loadCharacters();
  } catch (error) {
    console.error('Import failed:', error);
    alert('Failed to import character');
  }
}

// Media functionality
async function loadMediaList() {
  try {
    const response = await apiClient.getMediaList();
    displayMediaItems(response.media_items);
  } catch (error) {
    console.error('Failed to load media:', error);
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
    alert('Please enter a URL');
    return;
  }
  
  try {
    const response = await apiClient.processMediaUrl(url);
    alert('Media processed successfully!');
    loadMediaList();
  } catch (error) {
    console.error('Processing failed:', error);
    alert('Failed to process media');
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
    alert('File processed successfully!');
    loadMediaList();
  } catch (error) {
    console.error('Processing failed:', error);
    alert('Failed to process file');
  }
}

async function viewMediaItem(id) {
  try {
    const item = await apiClient.getMediaItem(id);
    // Open in new tab or display in modal
    console.log('Media item:', item);
    alert(`Viewing: ${item.title}`);
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
  }
}