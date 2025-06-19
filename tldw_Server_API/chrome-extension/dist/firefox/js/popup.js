// First thing - log that script is loading
console.log('popup.js starting to load');

// Browser API compatibility
const api = (typeof browser !== 'undefined') ? browser : chrome;
console.log('Browser API defined:', !!api);

// Create a global reference for consistency
window.browserAPI = api;

// Global state
let currentConversationId = null;
let selectedCharacterId = null;
let selectedPromptTemplate = null;

// Smart Context Detection System
class SmartContextDetector {
  constructor() {
    this.videoPatterns = [
      /youtube\.com\/watch/i,
      /youtu\.be\//i,
      /vimeo\.com\//i,
      /twitch\.tv\//i,
      /dailymotion\.com\//i,
      /facebook\.com\/.*\/videos\//i,
      /instagram\.com\/(p|tv|reel)\//i,
      /tiktok\.com\/@.*\/video\//i,
      /twitter\.com\/.*\/status\/.*\/video\//i,
      /reddit\.com\/r\/.*\/comments\/.*\/(v\.redd\.it|youtube)/i
    ];
    
    this.audioPatterns = [
      /spotify\.com\/(track|album|playlist|episode|show)/i,
      /soundcloud\.com\//i,
      /anchor\.fm\//i,
      /podcasts\.apple\.com\//i,
      /podcasts\.google\.com\//i,
      /overcast\.fm\//i,
      /castbox\.fm\//i,
      /stitcher\.com\//i
    ];
    
    this.articlePatterns = [
      /medium\.com\//i,
      /dev\.to\//i,
      /hashnode\./i,
      /substack\.com\//i,
      /github\.com\/.*\/blob\//i,
      /stackoverflow\.com\/questions\//i,
      /reddit\.com\/r\/.*\/comments\//i,
      /news\.ycombinator\.com\//i,
      /wikipedia\.org\/wiki\//i,
      /arxiv\.org\/abs\//i
    ];
    
    this.documentPatterns = [
      /\.pdf($|\?)/i,
      /docs\.google\.com\/(document|spreadsheets|presentation)/i,
      /drive\.google\.com\/file\/.*\/view/i,
      /dropbox\.com\/.*\.pdf/i,
      /onedrive\.live\.com\/.*\.pdf/i
    ];
  }
  
  detectContentType(url, title = '', pageContent = '') {
    const urlLower = url.toLowerCase();
    const titleLower = title.toLowerCase();
    const contentLower = pageContent.toLowerCase();
    
    // Check for video content
    if (this.videoPatterns.some(pattern => pattern.test(url))) {
      return {
        type: 'video',
        confidence: 0.9,
        suggestedAction: 'process-videos',
        icon: '🎥',
        description: 'Video content detected'
      };
    }
    
    // Check for audio content
    if (this.audioPatterns.some(pattern => pattern.test(url))) {
      return {
        type: 'audio',
        confidence: 0.9,
        suggestedAction: 'process-audios',
        icon: '🎵',
        description: 'Audio content detected'
      };
    }
    
    // Check for document content
    if (this.documentPatterns.some(pattern => pattern.test(url))) {
      return {
        type: 'document',
        confidence: 0.85,
        suggestedAction: 'process-documents',
        icon: '📄',
        description: 'Document content detected'
      };
    }
    
    // Check for article content
    if (this.articlePatterns.some(pattern => pattern.test(url))) {
      return {
        type: 'article',
        confidence: 0.8,
        suggestedAction: 'process-url',
        icon: '📰',
        description: 'Article content detected'
      };
    }
    
    // Advanced content analysis
    return this.analyzePageContent(url, title, pageContent);
  }
  
  analyzePageContent(url, title, pageContent) {
    let confidence = 0.5;
    let type = 'webpage';
    let suggestedAction = 'process-url';
    let icon = '🌐';
    let description = 'Web page content';
    
    // Check for code content
    if (this.hasCodeContent(url, title, pageContent)) {
      return {
        type: 'code',
        confidence: 0.75,
        suggestedAction: 'save-as-prompt',
        icon: '💻',
        description: 'Code content detected'
      };
    }
    
    // Check for long-form content
    if (pageContent.length > 3000) {
      confidence = 0.7;
      type = 'article';
      icon = '📖';
      description = 'Long-form content detected';
    }
    
    // Check for social media content
    if (this.isSocialMediaContent(url)) {
      return {
        type: 'social',
        confidence: 0.6,
        suggestedAction: 'send-to-chat',
        icon: '💬',
        description: 'Social media content'
      };
    }
    
    return { type, confidence, suggestedAction, icon, description };
  }
  
  hasCodeContent(url, title, content) {
    const codeIndicators = [
      /github\.com/i,
      /gitlab\.com/i,
      /bitbucket\.org/i,
      /codepen\.io/i,
      /jsfiddle\.net/i,
      /stackoverflow\.com/i,
      /\b(function|class|import|export|const|let|var)\b/i,
      /\b(def|class|import|from)\b/i, // Python
      /\b(public|private|protected|static)\b/i, // Java/C#
      /<\?php/i,
      /\{\s*\}/i,
      /\[\s*\]/i
    ];
    
    return codeIndicators.some(pattern => 
      pattern.test(url) || pattern.test(title) || pattern.test(content)
    );
  }
  
  isSocialMediaContent(url) {
    const socialPatterns = [
      /twitter\.com/i,
      /facebook\.com/i,
      /instagram\.com/i,
      /linkedin\.com/i,
      /tiktok\.com/i,
      /discord\.com/i,
      /telegram\.org/i
    ];
    
    return socialPatterns.some(pattern => pattern.test(url));
  }
  
  getSuggestedActions(context) {
    const actions = [];
    
    switch (context.type) {
      case 'video':
        actions.push(
          { label: 'Process Video', action: 'process-videos', primary: true },
          { label: 'Send to Chat', action: 'send-to-chat' },
          { label: 'Save URL', action: 'save-url' }
        );
        break;
        
      case 'audio':
        actions.push(
          { label: 'Process Audio', action: 'process-audios', primary: true },
          { label: 'Send to Chat', action: 'send-to-chat' },
          { label: 'Save URL', action: 'save-url' }
        );
        break;
        
      case 'document':
        actions.push(
          { label: 'Process Document', action: 'process-documents', primary: true },
          { label: 'Send to Chat', action: 'send-to-chat' }
        );
        break;
        
      case 'article':
        actions.push(
          { label: 'Process Article', action: 'process-url', primary: true },
          { label: 'Send to Chat', action: 'send-to-chat' },
          { label: 'Save as Prompt', action: 'save-as-prompt' }
        );
        break;
        
      case 'code':
        actions.push(
          { label: 'Save as Prompt', action: 'save-as-prompt', primary: true },
          { label: 'Send to Chat', action: 'send-to-chat' },
          { label: 'Process URL', action: 'process-url' }
        );
        break;
        
      case 'social':
        actions.push(
          { label: 'Send to Chat', action: 'send-to-chat', primary: true },
          { label: 'Save as Prompt', action: 'save-as-prompt' }
        );
        break;
        
      default:
        actions.push(
          { label: 'Process URL', action: 'process-url', primary: true },
          { label: 'Send to Chat', action: 'send-to-chat' },
          { label: 'Save as Prompt', action: 'save-as-prompt' }
        );
    }
    
    return actions;
  }
}

// Initialize smart context detector
const smartContext = new SmartContextDetector();

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
  console.log('DOMContentLoaded fired - popup.js is running');
  try {
    await initializePopup();
    setupEventListeners();
    setupTabs();
  } catch (error) {
    console.error('Error during initialization:', error);
    console.error('Stack trace:', error.stack);
  }
});

async function initializePopup() {
  console.log('Initializing popup...');
  
  // Initialize toast system
  toast.init();
  
  // Check connection status
  const isConnected = await apiClient.checkConnection();
  const status = apiClient.getConnectionStatus();
  updateConnectionStatus(isConnected, status);
  
  // Start connection monitoring
  startConnectionMonitoring();
  
  // Initialize smart context detection
  await initializeSmartContext();
  
  // Initialize enhanced search
  await enhancedSearch.initialize();
  
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
  
  console.log('Setting up tabs. Found buttons:', tabButtons.length, 'panes:', tabPanes.length);
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.dataset.tab;
      console.log('Tab clicked:', targetTab);
      
      // Update active states
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabPanes.forEach(pane => pane.classList.remove('active'));
      
      button.classList.add('active');
      const targetPane = document.getElementById(`${targetTab}-tab`);
      if (targetPane) {
        targetPane.classList.add('active');
      } else {
        console.error('Tab pane not found:', `${targetTab}-tab`);
      }
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
  
  // Batch operation events
  const processAllTabsBtn = document.getElementById('processAllTabs');
  const saveAllBookmarksBtn = document.getElementById('saveAllBookmarks');
  const processSelectedTabsBtn = document.getElementById('processSelectedTabs');
  
  if (processAllTabsBtn) {
    processAllTabsBtn.addEventListener('click', processAllTabs);
  }
  if (saveAllBookmarksBtn) {
    saveAllBookmarksBtn.addEventListener('click', saveAllBookmarks);
  }
  if (processSelectedTabsBtn) {
    processSelectedTabsBtn.addEventListener('click', processSelectedTabs);
  }
  
  // Footer events
  const optionsButton = document.getElementById('openOptions');
  const helpButton = document.getElementById('openHelp');
  
  if (optionsButton) {
    optionsButton.addEventListener('click', (e) => {
      e.preventDefault();
      console.log('Opening options page');
      api.runtime.openOptionsPage();
    });
  } else {
    console.error('Options button not found');
  }
  
  if (helpButton) {
    helpButton.addEventListener('click', (e) => {
      e.preventDefault();
      console.log('Opening help page');
      api.tabs.create({ url: 'https://github.com/rmusser01/tldw_server' });
    });
  } else {
    console.error('Help button not found');
  }
  
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
  
  const progressId = `chat-${Date.now()}`;
  const progress = progressIndicator.create(progressId, 'Sending Message', `Model: ${model}`, false);
  
  try {
    progress.update(20, 'Connecting to AI service...', 'Processing');
    
    const response = await apiClient.createChatCompletion(requestData);
    
    progress.update(80, 'Receiving response...', 'Almost done');
    
    if (response.choices && response.choices[0]) {
      const assistantMessage = response.choices[0].message.content;
      addMessageToChat('assistant', assistantMessage);
      
      // Update conversation ID if provided
      if (response.conversation_id) {
        currentConversationId = response.conversation_id;
      }
      
      progress.complete('Message sent successfully!');
    } else {
      progress.error('No response received from AI');
    }
  } catch (error) {
    console.error('Chat error:', error);
    addMessageToChat('system', `Error: ${error.message}`);
    progress.error(`Chat failed: ${error.message}`);
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

// Enhanced Search System
class EnhancedSearch {
  constructor() {
    this.recentSearches = {
      prompts: [],
      characters: []
    };
    this.searchCache = new Map();
    this.searchTimeout = null;
    this.maxRecentSearches = 10;
    
    this.loadRecentSearches();
  }
  
  async initialize() {
    this.setupSearchEventListeners();
    this.loadSearchData();
  }
  
  setupSearchEventListeners() {
    // Prompt search enhancements
    const promptSearch = document.getElementById('promptSearch');
    const promptSortBy = document.getElementById('promptSortBy');
    const promptFilterBy = document.getElementById('promptFilterBy');
    const clearPromptSearch = document.getElementById('clearPromptSearch');
    
    if (promptSearch) {
      promptSearch.addEventListener('input', (e) => this.handleSearchInput(e, 'prompts'));
      promptSearch.addEventListener('focus', () => this.showRecentSearches('prompts'));
      promptSearch.addEventListener('blur', () => {
        // Delay hiding to allow clicking on suggestions
        setTimeout(() => this.hideSearchSuggestions('prompts'), 200);
      });
    }
    
    if (promptSortBy) {
      promptSortBy.addEventListener('change', () => this.handleFilterChange('prompts'));
    }
    
    if (promptFilterBy) {
      promptFilterBy.addEventListener('change', () => this.handleFilterChange('prompts'));
    }
    
    if (clearPromptSearch) {
      clearPromptSearch.addEventListener('click', () => this.clearSearch('prompts'));
    }
    
    // Character search enhancements
    const characterSearch = document.getElementById('characterSearch');
    const characterSortBy = document.getElementById('characterSortBy');
    const characterFilterBy = document.getElementById('characterFilterBy');
    const clearCharacterSearch = document.getElementById('clearCharacterSearch');
    
    if (characterSearch) {
      characterSearch.addEventListener('input', (e) => this.handleSearchInput(e, 'characters'));
      characterSearch.addEventListener('focus', () => this.showRecentSearches('characters'));
      characterSearch.addEventListener('blur', () => {
        setTimeout(() => this.hideSearchSuggestions('characters'), 200);
      });
    }
    
    if (characterSortBy) {
      characterSortBy.addEventListener('change', () => this.handleFilterChange('characters'));
    }
    
    if (characterFilterBy) {
      characterFilterBy.addEventListener('change', () => this.handleFilterChange('characters'));
    }
    
    if (clearCharacterSearch) {
      clearCharacterSearch.addEventListener('click', () => this.clearSearch('characters'));
    }
  }
  
  handleSearchInput(event, type) {
    const query = event.target.value.trim();
    
    // Clear existing timeout
    if (this.searchTimeout) {
      clearTimeout(this.searchTimeout);
    }
    
    if (query.length === 0) {
      this.showRecentSearches(type);
      this.performSearch(type, '');
      return;
    }
    
    if (query.length >= 2) {
      this.showSearchSuggestions(type, query);
      
      // Debounced search
      this.searchTimeout = setTimeout(() => {
        this.performSearch(type, query);
      }, 300);
    }
  }
  
  async performSearch(type, query) {
    try {
      const filters = this.getSearchFilters(type);
      const cacheKey = `${type}-${query}-${JSON.stringify(filters)}`;
      
      // Check cache first
      if (this.searchCache.has(cacheKey)) {
        const results = this.searchCache.get(cacheKey);
        // Display cached results directly
        if (type === 'prompts') {
          displayPrompts(results);
        } else if (type === 'characters') {
          displayCharacters(results);
        }
        this.displaySearchStats(type, results, query);
        return;
      }
      
      let results;
      if (type === 'prompts') {
        if (query) {
          const response = await apiClient.searchPrompts(query);
          results = response.results;
        } else {
          results = await this.loadAllPrompts();
        }
        results = this.applySorting(results, filters.sortBy, type);
        displayPrompts(results);
      } else if (type === 'characters') {
        if (query) {
          const response = await apiClient.searchCharacters(query);
          results = response.results;
        } else {
          results = await this.loadAllCharacters();
        }
        results = this.applySorting(results, filters.sortBy, type);
        displayCharacters(results);
      }
      
      // Cache results
      this.searchCache.set(cacheKey, results);
      
      // Add to recent searches if query is not empty
      if (query) {
        this.addToRecentSearches(type, query);
      }
      
      this.displaySearchStats(type, results, query);
      
    } catch (error) {
      console.error(`${type} search failed:`, error);
      toast.error(`${type} search failed`);
    }
  }
  
  getSearchFilters(type) {
    const sortElement = document.getElementById(`${type.slice(0, -1)}SortBy`);
    const filterElement = document.getElementById(`${type.slice(0, -1)}FilterBy`);
    
    return {
      sortBy: sortElement ? sortElement.value : 'name',
      filterBy: filterElement ? filterElement.value : ''
    };
  }
  
  applySorting(results, sortBy, type) {
    if (!results || !Array.isArray(results)) return results;
    
    return results.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return (a.name || '').localeCompare(b.name || '');
        case 'created_at':
          return new Date(b.created_at || 0) - new Date(a.created_at || 0);
        case 'usage':
          return (b.usage_count || 0) - (a.usage_count || 0);
        case 'popularity':
          return (b.popularity || 0) - (a.popularity || 0);
        default:
          return 0;
      }
    });
  }
  
  async loadAllPrompts() {
    const response = await apiClient.getPrompts();
    return response.results || response;
  }
  
  async loadAllCharacters() {
    const response = await apiClient.getCharacters();
    return response.results || response;
  }
  
  showSearchSuggestions(type, query) {
    const suggestionsContainer = document.getElementById(`${type.slice(0, -1)}SearchSuggestions`);
    if (!suggestionsContainer) return;
    
    this.hideRecentSearches(type);
    
    const suggestions = this.generateSuggestions(type, query);
    
    if (suggestions.length === 0) {
      suggestionsContainer.style.display = 'none';
      return;
    }
    
    suggestionsContainer.innerHTML = suggestions.map(suggestion => `
      <div class="suggestion-item" data-suggestion="${suggestion.text}">
        ${this.highlightMatch(suggestion.text, query)}
        <span class="suggestion-type">${suggestion.type}</span>
      </div>
    `).join('');
    
    // Add click handlers
    suggestionsContainer.querySelectorAll('.suggestion-item').forEach(item => {
      item.addEventListener('click', () => {
        const searchInput = document.getElementById(`${type.slice(0, -1)}Search`);
        searchInput.value = item.dataset.suggestion;
        this.performSearch(type, item.dataset.suggestion);
        this.hideSearchSuggestions(type);
      });
    });
    
    suggestionsContainer.style.display = 'block';
  }
  
  generateSuggestions(type, query) {
    const suggestions = [];
    const queryLower = query.toLowerCase();
    
    // Common search terms for prompts
    if (type === 'prompts') {
      const promptSuggestions = [
        'writing', 'analysis', 'summary', 'creative', 'code review',
        'brainstorm', 'explain', 'translate', 'improve', 'debug'
      ];
      
      promptSuggestions.forEach(term => {
        if (term.includes(queryLower)) {
          suggestions.push({ text: term, type: 'common' });
        }
      });
    }
    
    // Common search terms for characters
    if (type === 'characters') {
      const characterSuggestions = [
        'assistant', 'expert', 'creative writer', 'analyst', 'teacher',
        'programmer', 'designer', 'consultant', 'researcher', 'mentor'
      ];
      
      characterSuggestions.forEach(term => {
        if (term.includes(queryLower)) {
          suggestions.push({ text: term, type: 'role' });
        }
      });
    }
    
    // Add recent searches that match
    this.recentSearches[type].forEach(recent => {
      if (recent.toLowerCase().includes(queryLower) && 
          !suggestions.find(s => s.text === recent)) {
        suggestions.push({ text: recent, type: 'recent' });
      }
    });
    
    return suggestions.slice(0, 8); // Limit to 8 suggestions
  }
  
  highlightMatch(text, query) {
    if (!query) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<span class="search-highlight">$1</span>');
  }
  
  showRecentSearches(type) {
    const recentContainer = document.getElementById(`${type.slice(0, -1)}RecentSearches`);
    const searchInput = document.getElementById(`${type.slice(0, -1)}Search`);
    
    if (!recentContainer || !searchInput || searchInput.value.trim()) return;
    
    this.hideSearchSuggestions(type);
    
    const recent = this.recentSearches[type];
    if (recent.length === 0) {
      recentContainer.style.display = 'none';
      return;
    }
    
    const recentItemsContainer = document.getElementById(`${type.slice(0, -1)}RecentItems`);
    recentItemsContainer.innerHTML = recent.map((search, index) => `
      <div class="recent-item" data-search="${search}">
        <span>${search}</span>
        <span class="recent-remove" data-index="${index}">×</span>
      </div>
    `).join('');
    
    // Add click handlers
    recentItemsContainer.querySelectorAll('.recent-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('recent-remove')) {
          const index = parseInt(e.target.dataset.index);
          this.removeRecentSearch(type, index);
          this.showRecentSearches(type);
        } else {
          searchInput.value = item.dataset.search;
          this.performSearch(type, item.dataset.search);
          this.hideRecentSearches(type);
        }
      });
    });
    
    recentContainer.style.display = 'block';
  }
  
  hideRecentSearches(type) {
    const recentContainer = document.getElementById(`${type.slice(0, -1)}RecentSearches`);
    if (recentContainer) {
      recentContainer.style.display = 'none';
    }
  }
  
  hideSearchSuggestions(type) {
    const suggestionsContainer = document.getElementById(`${type.slice(0, -1)}SearchSuggestions`);
    if (suggestionsContainer) {
      suggestionsContainer.style.display = 'none';
    }
  }
  
  addToRecentSearches(type, query) {
    const recent = this.recentSearches[type];
    
    // Remove if already exists
    const existingIndex = recent.indexOf(query);
    if (existingIndex !== -1) {
      recent.splice(existingIndex, 1);
    }
    
    // Add to beginning
    recent.unshift(query);
    
    // Limit to maxRecentSearches
    if (recent.length > this.maxRecentSearches) {
      recent.splice(this.maxRecentSearches);
    }
    
    this.saveRecentSearches();
  }
  
  removeRecentSearch(type, index) {
    this.recentSearches[type].splice(index, 1);
    this.saveRecentSearches();
  }
  
  clearSearch(type) {
    const searchInput = document.getElementById(`${type.slice(0, -1)}Search`);
    const sortSelect = document.getElementById(`${type.slice(0, -1)}SortBy`);
    const filterSelect = document.getElementById(`${type.slice(0, -1)}FilterBy`);
    
    if (searchInput) searchInput.value = '';
    if (sortSelect) sortSelect.value = 'name';
    if (filterSelect) filterSelect.value = '';
    
    this.hideSearchSuggestions(type);
    this.hideRecentSearches(type);
    
    this.performSearch(type, '');
  }
  
  handleFilterChange(type) {
    const searchInput = document.getElementById(`${type.slice(0, -1)}Search`);
    const query = searchInput ? searchInput.value.trim() : '';
    this.performSearch(type, query);
  }
  
  displaySearchStats(type, results, query) {
    const container = document.getElementById(`${type}List`);
    if (!container) return;
    
    let statsDiv = container.querySelector('.search-stats');
    if (!statsDiv) {
      statsDiv = document.createElement('div');
      statsDiv.className = 'search-stats';
      container.appendChild(statsDiv);
    }
    
    const count = results ? results.length : 0;
    if (query) {
      statsDiv.textContent = `Found ${count} result${count !== 1 ? 's' : ''} for "${query}"`;
    } else {
      statsDiv.textContent = `Showing ${count} ${type}`;
    }
  }
  
  loadRecentSearches() {
    const stored = localStorage.getItem('tldw-recent-searches');
    if (stored) {
      try {
        this.recentSearches = JSON.parse(stored);
      } catch (e) {
        console.warn('Failed to load recent searches:', e);
      }
    }
  }
  
  saveRecentSearches() {
    try {
      localStorage.setItem('tldw-recent-searches', JSON.stringify(this.recentSearches));
    } catch (e) {
      console.warn('Failed to save recent searches:', e);
    }
  }
  
  async loadSearchData() {
    // Pre-populate search cache with common data
    try {
      const [prompts, characters] = await Promise.all([
        this.loadAllPrompts(),
        this.loadAllCharacters()
      ]);
      
      this.searchCache.set('prompts--{"sortBy":"name","filterBy":""}', prompts);
      this.searchCache.set('characters--{"sortBy":"name","filterBy":""}', characters);
    } catch (error) {
      console.warn('Failed to preload search data:', error);
    }
  }
}

const enhancedSearch = new EnhancedSearch();

async function searchPrompts() {
  const query = document.getElementById('promptSearch').value.trim();
  await enhancedSearch.performSearch('prompts', query);
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
    
    // Use api for downloads
    if (api.downloads) {
      api.downloads.download({
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
  await enhancedSearch.performSearch('characters', query);
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

// Progress Indicator System
class ProgressIndicator {
  constructor() {
    this.activeOperations = new Map();
    this.setupProgressContainer();
  }
  
  setupProgressContainer() {
    // Create global progress container if it doesn't exist
    if (!document.getElementById('global-progress-container')) {
      const container = document.createElement('div');
      container.id = 'global-progress-container';
      container.className = 'global-progress-container';
      document.body.appendChild(container);
      
      // Add CSS
      this.addProgressCSS();
    }
  }
  
  addProgressCSS() {
    if (document.getElementById('progress-css')) return;
    
    const style = document.createElement('style');
    style.id = 'progress-css';
    style.textContent = `
      .global-progress-container {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9999;
        max-width: 300px;
      }
      
      .progress-item {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        overflow: hidden;
        animation: slideIn 0.3s ease;
      }
      
      .progress-header {
        padding: 12px;
        background: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      
      .progress-title {
        font-size: 13px;
        font-weight: 600;
        color: #495057;
      }
      
      .progress-close {
        background: none;
        border: none;
        color: #6c757d;
        cursor: pointer;
        font-size: 16px;
        padding: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      
      .progress-close:hover {
        color: #dc3545;
      }
      
      .progress-body {
        padding: 12px;
      }
      
      .progress-description {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 8px;
      }
      
      .progress-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 6px;
        overflow: hidden;
        margin-bottom: 8px;
      }
      
      .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
        transition: width 0.3s ease;
        border-radius: 10px;
      }
      
      .progress-bar-fill.indeterminate {
        width: 30% !important;
        animation: indeterminate 2s infinite linear;
      }
      
      .progress-stats {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #6c757d;
      }
      
      .progress-status {
        font-weight: 500;
      }
      
      .progress-eta {
        font-style: italic;
      }
      
      @keyframes slideIn {
        from {
          opacity: 0;
          transform: translateX(100%);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }
      
      @keyframes indeterminate {
        0% {
          transform: translateX(-100%);
        }
        100% {
          transform: translateX(300%);
        }
      }
      
      .upload-progress {
        margin-top: 8px;
      }
      
      .upload-progress .progress-file {
        font-size: 12px;
        color: #495057;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
      }
      
      .upload-progress .progress-speed {
        font-size: 11px;
        color: #6c757d;
      }
    `;
    
    document.head.appendChild(style);
  }
  
  create(id, title, description = '', canCancel = false) {
    const operation = {
      id,
      title,
      description,
      startTime: Date.now(),
      progress: 0,
      status: 'starting',
      canCancel,
      cancelled: false
    };
    
    this.activeOperations.set(id, operation);
    this.renderProgressItem(operation);
    
    return {
      update: (progress, description, status) => this.update(id, progress, description, status),
      complete: (message) => this.complete(id, message),
      error: (error) => this.error(id, error),
      cancel: () => this.cancel(id)
    };
  }
  
  renderProgressItem(operation) {
    const container = document.getElementById('global-progress-container');
    
    const progressItem = document.createElement('div');
    progressItem.id = `progress-${operation.id}`;
    progressItem.className = 'progress-item';
    
    progressItem.innerHTML = `
      <div class="progress-header">
        <div class="progress-title">${operation.title}</div>
        ${operation.canCancel ? `<button class="progress-close" onclick="progressIndicator.cancel('${operation.id}')">×</button>` : ''}
      </div>
      <div class="progress-body">
        <div class="progress-description">${operation.description}</div>
        <div class="progress-bar-container">
          <div class="progress-bar-fill ${operation.progress === 0 ? 'indeterminate' : ''}" style="width: ${operation.progress}%"></div>
        </div>
        <div class="progress-stats">
          <span class="progress-status">${operation.status}</span>
          <span class="progress-eta"></span>
        </div>
      </div>
    `;
    
    container.appendChild(progressItem);
  }
  
  update(id, progress, description, status) {
    const operation = this.activeOperations.get(id);
    if (!operation || operation.cancelled) return;
    
    operation.progress = Math.max(0, Math.min(100, progress));
    if (description) operation.description = description;
    if (status) operation.status = status;
    
    const progressItem = document.getElementById(`progress-${id}`);
    if (!progressItem) return;
    
    const progressFill = progressItem.querySelector('.progress-bar-fill');
    const progressDesc = progressItem.querySelector('.progress-description');
    const progressStatus = progressItem.querySelector('.progress-status');
    const progressEta = progressItem.querySelector('.progress-eta');
    
    if (progressFill) {
      progressFill.style.width = `${operation.progress}%`;
      if (operation.progress > 0) {
        progressFill.classList.remove('indeterminate');
      }
    }
    
    if (progressDesc && description) {
      progressDesc.textContent = description;
    }
    
    if (progressStatus && status) {
      progressStatus.textContent = status;
    }
    
    if (progressEta && operation.progress > 0 && operation.progress < 100) {
      const elapsed = Date.now() - operation.startTime;
      const estimated = (elapsed / operation.progress) * (100 - operation.progress);
      const etaSeconds = Math.round(estimated / 1000);
      
      if (etaSeconds > 0) {
        progressEta.textContent = `ETA: ${this.formatTime(etaSeconds)}`;
      }
    }
  }
  
  complete(id, message = 'Completed successfully') {
    const operation = this.activeOperations.get(id);
    if (!operation) return;
    
    this.update(id, 100, message, 'Completed');
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
      this.remove(id);
    }, 3000);
  }
  
  error(id, error = 'Operation failed') {
    const operation = this.activeOperations.get(id);
    if (!operation) return;
    
    const progressItem = document.getElementById(`progress-${id}`);
    if (progressItem) {
      progressItem.style.borderColor = '#dc3545';
      progressItem.querySelector('.progress-bar-fill').style.background = '#dc3545';
      this.update(id, operation.progress, error, 'Error');
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      this.remove(id);
    }, 5000);
  }
  
  cancel(id) {
    const operation = this.activeOperations.get(id);
    if (!operation) return;
    
    operation.cancelled = true;
    this.update(id, operation.progress, 'Operation cancelled', 'Cancelled');
    
    // Auto-remove after 2 seconds
    setTimeout(() => {
      this.remove(id);
    }, 2000);
    
    // Trigger cancel event if callback exists
    if (operation.onCancel) {
      operation.onCancel();
    }
  }
  
  remove(id) {
    const progressItem = document.getElementById(`progress-${id}`);
    if (progressItem) {
      progressItem.style.animation = 'slideIn 0.3s ease reverse';
      setTimeout(() => {
        progressItem.remove();
      }, 300);
    }
    
    this.activeOperations.delete(id);
  }
  
  formatTime(seconds) {
    if (seconds < 60) {
      return `${seconds}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  }
  
  // Helper method for file upload progress
  createUploadProgress(id, file) {
    const progress = this.create(id, `Uploading ${file.name}`, `File size: ${this.formatFileSize(file.size)}`, true);
    
    return {
      ...progress,
      updateUpload: (loaded, total, speed) => {
        const percentage = (loaded / total) * 100;
        const description = `${this.formatFileSize(loaded)} / ${this.formatFileSize(total)}`;
        const status = speed ? `Upload speed: ${this.formatFileSize(speed)}/s` : 'Uploading...';
        
        this.update(id, percentage, description, status);
      }
    };
  }
  
  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}

const progressIndicator = new ProgressIndicator();

async function processMediaUrl() {
  const url = document.getElementById('mediaUrl').value.trim();
  if (!url) {
    toast.warning('Please enter a URL');
    return;
  }
  
  const progressId = `url-${Date.now()}`;
  const progress = progressIndicator.create(progressId, 'Processing URL', `Processing: ${url}`, true);
  
  try {
    progress.update(25, 'Connecting to server...', 'Processing');
    
    const response = await apiClient.processMediaUrl(url);
    
    progress.update(75, 'Finalizing...', 'Almost done');
    
    // Simulate processing time for better UX
    await new Promise(resolve => setTimeout(resolve, 500));
    
    progress.complete('Media processed successfully!');
    toast.success('Media processed successfully!');
    loadMediaList();
  } catch (error) {
    console.error('Processing failed:', error);
    progress.error(`Failed to process: ${error.message}`);
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
  
  const progressId = `file-${Date.now()}`;
  const progress = progressIndicator.createUploadProgress(progressId, file);
  
  try {
    // Simulate upload progress since we can't track real upload progress with basic fetch
    let uploadProgress = 0;
    const uploadInterval = setInterval(() => {
      uploadProgress += Math.random() * 20;
      if (uploadProgress >= 90) {
        clearInterval(uploadInterval);
        progress.update(90, 'Processing file...', 'Processing');
        return;
      }
      progress.updateUpload(uploadProgress * file.size / 100, file.size, null);
    }, 200);
    
    const response = await apiClient.processMediaFile(file, endpoint);
    
    clearInterval(uploadInterval);
    progress.complete('File processed successfully!');
    toast.success('File processed successfully!');
    loadMediaList();
  } catch (error) {
    console.error('Processing failed:', error);
    progress.error(`Failed to process: ${error.message}`);
    toast.error(`Failed to process file: ${error.message}`);
  }
}

// Batch Operations
class BatchProcessor {
  constructor() {
    this.isProcessing = false;
    this.currentBatch = [];
    this.progressCallback = null;
  }
  
  async processAllTabs() {
    try {
      const tabs = await api.tabs.query({ currentWindow: true });
      const filteredTabs = tabs.filter(tab => !tab.url.startsWith('chrome://') && !tab.url.startsWith('moz-extension://'));
      
      if (filteredTabs.length === 0) {
        toast.warning('No processable tabs found');
        return;
      }
      
      const confirmed = await this.confirmBatchOperation(
        'Process All Tabs',
        `Process ${filteredTabs.length} tabs? This may take several minutes.`
      );
      
      if (!confirmed) return;
      
      await this.processTabs(filteredTabs);
    } catch (error) {
      console.error('Failed to process all tabs:', error);
      toast.error(`Failed to process tabs: ${error.message}`);
    }
  }
  
  async processSelectedTabs() {
    try {
      const tabs = await api.tabs.query({ currentWindow: true });
      const filteredTabs = tabs.filter(tab => !tab.url.startsWith('chrome://') && !tab.url.startsWith('moz-extension://'));
      
      if (filteredTabs.length === 0) {
        toast.warning('No processable tabs found');
        return;
      }
      
      const selectedTabs = await this.showTabSelectionModal(filteredTabs);
      if (!selectedTabs || selectedTabs.length === 0) return;
      
      await this.processTabs(selectedTabs);
    } catch (error) {
      console.error('Failed to process selected tabs:', error);
      toast.error(`Failed to process tabs: ${error.message}`);
    }
  }
  
  async saveAllBookmarks() {
    try {
      // Check if bookmarks permission is available
      if (!api.bookmarks) {
        toast.error('Bookmarks permission not granted. Please enable it in extension settings.');
        return;
      }
      
      const bookmarks = await this.getAllBookmarks();
      const processableBookmarks = bookmarks.filter(bm => 
        bm.url && !bm.url.startsWith('chrome://') && !bm.url.startsWith('moz-extension://')
      );
      
      if (processableBookmarks.length === 0) {
        toast.warning('No processable bookmarks found');
        return;
      }
      
      const confirmed = await this.confirmBatchOperation(
        'Save All Bookmarks',
        `Process ${processableBookmarks.length} bookmarks? This may take several minutes.`
      );
      
      if (!confirmed) return;
      
      await this.processBookmarks(processableBookmarks);
    } catch (error) {
      console.error('Failed to save bookmarks:', error);
      toast.error(`Failed to save bookmarks: ${error.message}`);
    }
  }
  
  async processTabs(tabs) {
    this.isProcessing = true;
    this.showProgress(true);
    
    let completed = 0;
    const total = tabs.length;
    
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      this.updateProgress(completed, total, `Processing: ${tab.title}`);
      
      try {
        // Detect content type for each tab
        const context = smartContext.detectContentType(tab.url, tab.title);
        const endpoint = this.getEndpointForContext(context);
        
        await apiClient.processUrl(tab.url, endpoint);
        completed++;
        
        // Small delay to prevent overwhelming the server
        await this.delay(1000);
      } catch (error) {
        console.error(`Failed to process tab: ${tab.title}`, error);
      }
      
      this.updateProgress(completed, total);
    }
    
    this.showProgress(false);
    this.isProcessing = false;
    
    toast.success(`Batch processing completed! ${completed}/${total} tabs processed successfully.`);
    loadMediaList();
  }
  
  async processBookmarks(bookmarks) {
    this.isProcessing = true;
    this.showProgress(true);
    
    let completed = 0;
    const total = bookmarks.length;
    
    for (let i = 0; i < bookmarks.length; i++) {
      const bookmark = bookmarks[i];
      this.updateProgress(completed, total, `Processing: ${bookmark.title}`);
      
      try {
        const context = smartContext.detectContentType(bookmark.url, bookmark.title);
        const endpoint = this.getEndpointForContext(context);
        
        await apiClient.processUrl(bookmark.url, endpoint);
        completed++;
        
        await this.delay(1500); // Longer delay for bookmarks
      } catch (error) {
        console.error(`Failed to process bookmark: ${bookmark.title}`, error);
      }
      
      this.updateProgress(completed, total);
    }
    
    this.showProgress(false);
    this.isProcessing = false;
    
    toast.success(`Bookmark processing completed! ${completed}/${total} bookmarks processed successfully.`);
    loadMediaList();
  }
  
  getEndpointForContext(context) {
    switch (context.type) {
      case 'video': return 'process-videos';
      case 'audio': return 'process-audios';
      case 'document': return 'process-documents';
      default: return 'process-url';
    }
  }
  
  async getAllBookmarks() {
    const getAllBookmarksRecursive = async (node) => {
      let bookmarks = [];
      
      if (node.children) {
        for (const child of node.children) {
          bookmarks = bookmarks.concat(await getAllBookmarksRecursive(child));
        }
      } else if (node.url) {
        bookmarks.push(node);
      }
      
      return bookmarks;
    };
    
    const bookmarkTree = await api.bookmarks.getTree();
    return getAllBookmarksRecursive(bookmarkTree[0]);
  }
  
  async confirmBatchOperation(title, message) {
    return new Promise((resolve) => {
      const modal = this.createConfirmModal(title, message, resolve);
      document.body.appendChild(modal);
    });
  }
  
  createConfirmModal(title, message, callback) {
    const modal = document.createElement('div');
    modal.className = 'tab-selection-modal';
    modal.innerHTML = `
      <div class="tab-selection-content">
        <div class="tab-selection-header">
          <h3>${title}</h3>
          <button class="modal-close" onclick="this.closest('.tab-selection-modal').remove(); arguments[0](false);">×</button>
        </div>
        <p style="margin-bottom: 20px;">${message}</p>
        <div class="tab-selection-actions">
          <button class="btn btn-secondary" onclick="this.closest('.tab-selection-modal').remove(); arguments[0](false);">Cancel</button>
          <button class="btn btn-primary" onclick="this.closest('.tab-selection-modal').remove(); arguments[0](true);">Confirm</button>
        </div>
      </div>
    `;
    
    // Set up event handlers with callback
    modal.querySelector('.btn.btn-secondary').onclick = () => {
      modal.remove();
      callback(false);
    };
    modal.querySelector('.btn.btn-primary').onclick = () => {
      modal.remove();
      callback(true);
    };
    modal.querySelector('.modal-close').onclick = () => {
      modal.remove();
      callback(false);
    };
    
    return modal;
  }
  
  async showTabSelectionModal(tabs) {
    return new Promise((resolve) => {
      const modal = this.createTabSelectionModal(tabs, resolve);
      document.body.appendChild(modal);
    });
  }
  
  createTabSelectionModal(tabs, callback) {
    const modal = document.createElement('div');
    modal.className = 'tab-selection-modal';
    
    const tabItems = tabs.map(tab => `
      <div class="tab-item" data-tab-id="${tab.id}">
        <input type="checkbox" class="tab-checkbox" />
        <div class="tab-info">
          <div class="tab-title">${tab.title}</div>
          <div class="tab-url">${tab.url}</div>
        </div>
      </div>
    `).join('');
    
    modal.innerHTML = `
      <div class="tab-selection-content">
        <div class="tab-selection-header">
          <h3>Select Tabs to Process</h3>
          <button class="modal-close">×</button>
        </div>
        <div class="tab-list" style="max-height: 300px; overflow-y: auto;">
          ${tabItems}
        </div>
        <div class="tab-selection-actions">
          <button class="btn btn-secondary select-all-btn">Select All</button>
          <button class="btn btn-secondary">Cancel</button>
          <button class="btn btn-primary">Process Selected</button>
        </div>
      </div>
    `;
    
    // Set up event handlers
    modal.querySelector('.select-all-btn').onclick = () => {
      const checkboxes = modal.querySelectorAll('.tab-checkbox');
      const allChecked = Array.from(checkboxes).every(cb => cb.checked);
      checkboxes.forEach(cb => cb.checked = !allChecked);
    };
    
    modal.querySelector('.btn.btn-secondary:not(.select-all-btn)').onclick = () => {
      modal.remove();
      callback(null);
    };
    
    modal.querySelector('.btn.btn-primary').onclick = () => {
      const selectedTabIds = Array.from(modal.querySelectorAll('.tab-item'))
        .filter(item => item.querySelector('.tab-checkbox').checked)
        .map(item => parseInt(item.dataset.tabId));
      
      const selectedTabs = tabs.filter(tab => selectedTabIds.includes(tab.id));
      modal.remove();
      callback(selectedTabs);
    };
    
    modal.querySelector('.modal-close').onclick = () => {
      modal.remove();
      callback(null);
    };
    
    // Make tab items clickable
    modal.querySelectorAll('.tab-item').forEach(item => {
      item.onclick = (e) => {
        if (e.target.type !== 'checkbox') {
          const checkbox = item.querySelector('.tab-checkbox');
          checkbox.checked = !checkbox.checked;
        }
        item.classList.toggle('selected', item.querySelector('.tab-checkbox').checked);
      };
    });
    
    return modal;
  }
  
  showProgress(show) {
    const progressDiv = document.getElementById('batchProgress');
    if (progressDiv) {
      progressDiv.style.display = show ? 'block' : 'none';
    }
    
    // Disable batch buttons during processing
    const batchButtons = document.querySelectorAll('.batch-btn');
    batchButtons.forEach(btn => btn.disabled = show);
  }
  
  updateProgress(completed, total, currentItem = '') {
    const progressFill = document.getElementById('batchProgressFill');
    const progressText = document.getElementById('batchProgressText');
    
    if (progressFill && progressText) {
      const percentage = total > 0 ? (completed / total) * 100 : 0;
      progressFill.style.width = `${percentage}%`;
      
      let text = `${completed}/${total} completed`;
      if (currentItem) {
        text += ` - ${currentItem}`;
      }
      progressText.textContent = text;
    }
  }
  
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

const batchProcessor = new BatchProcessor();

// Batch operation handlers
async function processAllTabs() {
  if (batchProcessor.isProcessing) {
    toast.warning('Batch operation already in progress');
    return;
  }
  await batchProcessor.processAllTabs();
}

async function processSelectedTabs() {
  if (batchProcessor.isProcessing) {
    toast.warning('Batch operation already in progress');
    return;
  }
  await batchProcessor.processSelectedTabs();
}

async function saveAllBookmarks() {
  if (batchProcessor.isProcessing) {
    toast.warning('Batch operation already in progress');
    return;
  }
  await batchProcessor.saveAllBookmarks();
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

// Smart Context Detection UI
async function initializeSmartContext() {
  try {
    // Get current tab information
    const [tab] = await api.tabs.query({ active: true, currentWindow: true });
    if (!tab) return;
    
    // Get page content from content script
    let pageInfo = { title: tab.title, url: tab.url, pageContent: '', selectedText: '' };
    
    try {
      const response = await api.tabs.sendMessage(tab.id, { action: 'getPageInfo' });
      if (response) {
        pageInfo = { ...pageInfo, ...response };
      }
    } catch (error) {
      console.log('Could not get page info from content script:', error);
    }
    
    // Detect content type
    const context = smartContext.detectContentType(pageInfo.url, pageInfo.title, pageInfo.pageContent);
    
    // Update UI with smart suggestions
    updateSmartContextUI(context, pageInfo);
    
  } catch (error) {
    console.error('Failed to initialize smart context:', error);
  }
}

function updateSmartContextUI(context, pageInfo) {
  // Find or create smart context container
  let contextContainer = document.getElementById('smart-context');
  if (!contextContainer) {
    contextContainer = document.createElement('div');
    contextContainer.id = 'smart-context';
    contextContainer.className = 'smart-context-container';
    
    // Insert at the top of the main content area
    const mainContent = document.querySelector('.popup-content');
    if (mainContent && mainContent.firstChild) {
      mainContent.insertBefore(contextContainer, mainContent.firstChild);
    }
  }
  
  // Only show if confidence is reasonable
  if (context.confidence < 0.6) {
    contextContainer.style.display = 'none';
    return;
  }
  
  contextContainer.style.display = 'block';
  
  // Create context info display
  contextContainer.innerHTML = `
    <div class="context-header">
      <span class="context-icon">${context.icon}</span>
      <div class="context-info">
        <div class="context-type">${context.description}</div>
        <div class="context-confidence">Confidence: ${Math.round(context.confidence * 100)}%</div>
      </div>
    </div>
    <div class="context-actions" id="context-actions">
      <!-- Actions will be populated here -->
    </div>
  `;
  
  // Add suggested actions
  const actionsContainer = document.getElementById('context-actions');
  const suggestedActions = smartContext.getSuggestedActions(context);
  
  suggestedActions.forEach((action, index) => {
    const button = document.createElement('button');
    button.className = `context-action-btn ${action.primary ? 'primary' : 'secondary'}`;
    button.textContent = action.label;
    button.onclick = () => executeContextAction(action.action, pageInfo, context);
    
    actionsContainer.appendChild(button);
  });
  
  // Add CSS if not already present
  addSmartContextCSS();
}

function addSmartContextCSS() {
  if (document.getElementById('smart-context-css')) return;
  
  const style = document.createElement('style');
  style.id = 'smart-context-css';
  style.textContent = `
    .smart-context-container {
      background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
      border: 1px solid #dee2e6;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 16px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .context-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }
    
    .context-icon {
      font-size: 20px;
    }
    
    .context-info {
      flex: 1;
    }
    
    .context-type {
      font-weight: 600;
      font-size: 14px;
      color: #343a40;
    }
    
    .context-confidence {
      font-size: 12px;
      color: #6c757d;
    }
    
    .context-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    
    .context-action-btn {
      padding: 6px 12px;
      border: 1px solid #ddd;
      border-radius: 4px;
      background: white;
      cursor: pointer;
      font-size: 12px;
      transition: all 0.2s ease;
    }
    
    .context-action-btn.primary {
      background: #007bff;
      color: white;
      border-color: #007bff;
    }
    
    .context-action-btn.primary:hover {
      background: #0056b3;
      border-color: #0056b3;
    }
    
    .context-action-btn.secondary:hover {
      background: #f8f9fa;
      border-color: #adb5bd;
    }
  `;
  
  document.head.appendChild(style);
}

async function executeContextAction(action, pageInfo, context) {
  try {
    switch (action) {
      case 'process-videos':
        await processCurrentUrl('process-videos');
        break;
      case 'process-audios':
        await processCurrentUrl('process-audios');
        break;
      case 'process-documents':
        await processCurrentUrl('process-documents');
        break;
      case 'process-url':
        await processCurrentUrl('process-url');
        break;
      case 'send-to-chat':
        sendPageToChat(pageInfo);
        break;
      case 'save-as-prompt':
        openPromptModalWithContent(pageInfo);
        break;
      case 'save-url':
        saveCurrentUrl();
        break;
      default:
        toast.info(`Action "${action}" not implemented yet`);
    }
  } catch (error) {
    console.error('Context action failed:', error);
    toast.error(`Failed to execute ${action}: ${error.message}`);
  }
}

function sendPageToChat(pageInfo) {
  // Switch to chat tab
  document.querySelector('[data-tab="chat"]').click();
  
  // Populate chat input
  const chatInput = document.getElementById('chatInput');
  const content = pageInfo.selectedText || `Content from: ${pageInfo.title}\nURL: ${pageInfo.url}`;
  chatInput.value = content;
  
  toast.success('Content added to chat');
}

function openPromptModalWithContent(pageInfo) {
  openPromptModal();
  
  // Pre-populate with page content
  const titleInput = document.getElementById('prompt-title');
  const contentInput = document.getElementById('prompt-content');
  
  if (titleInput && contentInput) {
    titleInput.value = `Content from ${pageInfo.title}`;
    contentInput.value = pageInfo.selectedText || pageInfo.pageContent.substring(0, 1000);
  }
}

async function processCurrentUrl(endpoint) {
  try {
    const [tab] = await api.tabs.query({ active: true, currentWindow: true });
    if (!tab) throw new Error('No active tab found');
    
    const response = await apiClient.processUrl(tab.url, endpoint);
    toast.success(`Processing started for ${tab.title}`);
    
    // Refresh media list
    setTimeout(() => loadMediaList(), 2000);
    
  } catch (error) {
    console.error('Failed to process URL:', error);
    toast.error(`Failed to process URL: ${error.message}`);
  }
}

function saveCurrentUrl() {
  toast.info('URL save functionality coming soon!');
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