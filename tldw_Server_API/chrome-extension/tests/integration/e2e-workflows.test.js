/**
 * @jest-environment jsdom
 */

import { 
  createMockPrompt, 
  createMockCharacter, 
  createMockChatResponse,
  createMockMediaItem 
} from '../utils/factories.js';
import { 
  waitForAsync, 
  setupMockAPI, 
  setupChromeStorage,
  simulateUserInput,
  waitForElement,
  createDOM,
  cleanupDOM
} from '../utils/helpers.js';
import '../utils/matchers.js';

// Mock all required modules
jest.mock('../../js/utils/api.js');
jest.mock('../../js/utils/config.js');

describe('End-to-End Workflow Tests', () => {
  let apiClient;
  let configManager;
  let mockStorage;

  beforeEach(async () => {
    // Setup mocks
    const apiModule = require('../../js/utils/api.js');
    const configModule = require('../../js/utils/config.js');
    
    apiClient = {
      checkConnection: jest.fn().mockResolvedValue(true),
      createChatCompletion: jest.fn(),
      createPrompt: jest.fn(),
      searchPrompts: jest.fn(),
      getPrompts: jest.fn(),
      getCharacters: jest.fn(),
      selectCharacter: jest.fn(),
      processUrl: jest.fn(),
      getMediaList: jest.fn(),
      exportPrompts: jest.fn()
    };
    
    configManager = {
      loadConfig: jest.fn().mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiKey: null,
        defaultModel: 'gpt-3.5-turbo'
      }),
      saveConfig: jest.fn().mockResolvedValue(true),
      getConfig: jest.fn().mockReturnValue({
        serverUrl: 'http://localhost:8000',
        apiKey: null,
        defaultModel: 'gpt-3.5-turbo'
      })
    };
    
    apiModule.apiClient = apiClient;
    configModule.configManager = configManager;
    
    // Setup Chrome storage
    mockStorage = setupChromeStorage({
      sync: {
        config: {
          serverUrl: 'http://localhost:8000',
          apiKey: null
        }
      },
      local: {
        recentSearches: {
          prompts: [],
          characters: []
        },
        currentConversation: []
      }
    });

    // Setup DOM
    document.body.innerHTML = await fetch('/html/popup.html')
      .then(r => r.text())
      .catch(() => `
        <div id="toast-container"></div>
        <div class="container">
          <div class="tabs">
            <button class="tab-button active" data-tab="chat">Chat</button>
            <button class="tab-button" data-tab="prompts">Prompts</button>
            <button class="tab-button" data-tab="characters">Characters</button>
            <button class="tab-button" data-tab="media">Media</button>
          </div>
          
          <div class="tab-content">
            <div id="chat-tab" class="tab-pane active">
              <select id="modelSelect">
                <option value="gpt-3.5-turbo">GPT-3.5</option>
                <option value="gpt-4">GPT-4</option>
              </select>
              <select id="characterSelect">
                <option value="">No Character</option>
              </select>
              <div id="chatMessages"></div>
              <textarea id="chatInput"></textarea>
              <button id="sendMessage">Send</button>
              <button id="clearChat">Clear</button>
            </div>
            
            <div id="prompts-tab" class="tab-pane">
              <input id="promptSearch" type="text" />
              <button id="searchPromptsBtn">Search</button>
              <button id="createPrompt">Create New</button>
              <div id="promptsList"></div>
            </div>
            
            <div id="characters-tab" class="tab-pane">
              <input id="characterSearch" type="text" />
              <button id="searchCharactersBtn">Search</button>
              <div id="charactersList"></div>
            </div>
            
            <div id="media-tab" class="tab-pane">
              <input id="mediaUrl" type="text" />
              <button id="processUrl">Process URL</button>
              <div id="mediaList"></div>
            </div>
          </div>
          
          <div id="connectionStatus" class="status-indicator">
            <span id="connectionText">Connected</span>
          </div>
        </div>
        
        <!-- Modals -->
        <div id="prompt-modal" style="display: none;">
          <div class="modal-content">
            <h3>Create New Prompt</h3>
            <form id="prompt-form">
              <input id="prompt-name" placeholder="Prompt Name" required />
              <textarea id="prompt-content" placeholder="Prompt Content" required></textarea>
              <input id="prompt-keywords" placeholder="Keywords (comma-separated)" />
              <button type="submit" id="save-prompt">Save</button>
              <button type="button" id="cancel-prompt">Cancel</button>
            </form>
          </div>
        </div>
      `);

    // Initialize basic event listeners
    jest.resetModules();
  });

  afterEach(() => {
    cleanupDOM();
    jest.clearAllMocks();
  });

  describe('Complete User Journey: First Time Setup', () => {
    test('should guide user through initial configuration', async () => {
      // Simulate extension install
      apiClient.checkConnection.mockResolvedValueOnce(false);
      
      // User opens popup for first time
      const popup = require('../../js/popup.js');
      await waitForAsync();
      
      // Should show disconnected status
      expect(document.getElementById('connectionText').textContent).toContain('Disconnected');
      
      // User goes to settings (would normally open options page)
      expect(chrome.runtime.openOptionsPage).toBeDefined();
      
      // Simulate configuration save
      configManager.saveConfig.mockResolvedValueOnce(true);
      apiClient.checkConnection.mockResolvedValueOnce(true);
      
      // After configuration, should connect
      await waitForAsync();
      
      // Verify connection established
      expect(apiClient.checkConnection).toHaveBeenCalled();
    });
  });

  describe('Chat Workflow with Character Selection', () => {
    beforeEach(() => {
      const mockCharacters = [
        createMockCharacter({ id: '1', name: 'Assistant' }),
        createMockCharacter({ id: '2', name: 'Creative Writer' })
      ];
      
      apiClient.getCharacters.mockResolvedValue({ results: mockCharacters });
      apiClient.createChatCompletion.mockResolvedValue(
        createMockChatResponse({ 
          choices: [{ 
            message: { 
              role: 'assistant', 
              content: 'Hello! How can I help you today?' 
            } 
          }] 
        })
      );
    });

    test('should complete full chat interaction with character', async () => {
      // Load characters
      await apiClient.getCharacters();
      const characterSelect = document.getElementById('characterSelect');
      
      // Populate character options
      const characters = await apiClient.getCharacters();
      characters.results.forEach(char => {
        const option = document.createElement('option');
        option.value = char.id;
        option.textContent = char.name;
        characterSelect.appendChild(option);
      });
      
      // Select a character
      characterSelect.value = '2';
      characterSelect.dispatchEvent(new Event('change'));
      
      // Type a message
      const chatInput = document.getElementById('chatInput');
      simulateUserInput(chatInput, 'Write me a short story');
      
      // Send message
      const sendButton = document.getElementById('sendMessage');
      sendButton.click();
      
      await waitForAsync();
      
      // Verify API call
      expect(apiClient.createChatCompletion).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            expect.objectContaining({
              role: 'user',
              content: 'Write me a short story'
            })
          ]),
          character_id: '2'
        })
      );
      
      // Check response displayed
      const chatMessages = document.getElementById('chatMessages');
      expect(chatMessages.innerHTML).toContain('Write me a short story');
      expect(chatMessages.innerHTML).toContain('Hello! How can I help you today?');
    });

    test('should maintain conversation context', async () => {
      // First message
      const chatInput = document.getElementById('chatInput');
      simulateUserInput(chatInput, 'Hello');
      document.getElementById('sendMessage').click();
      
      await waitForAsync();
      
      // Second message
      apiClient.createChatCompletion.mockResolvedValue(
        createMockChatResponse({ 
          choices: [{ 
            message: { 
              role: 'assistant', 
              content: 'I can help you with that!' 
            } 
          }] 
        })
      );
      
      simulateUserInput(chatInput, 'Can you help me?');
      document.getElementById('sendMessage').click();
      
      await waitForAsync();
      
      // Should include conversation history
      expect(apiClient.createChatCompletion).toHaveBeenLastCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            expect.objectContaining({ role: 'user', content: 'Hello' }),
            expect.objectContaining({ role: 'assistant', content: 'Hello! How can I help you today?' }),
            expect.objectContaining({ role: 'user', content: 'Can you help me?' })
          ])
        })
      );
    });
  });

  describe('Prompt Creation and Search Workflow', () => {
    const mockPrompts = [
      createMockPrompt({ name: 'Email Writer', keywords: ['email', 'business'] }),
      createMockPrompt({ name: 'Code Review', keywords: ['code', 'programming'] }),
      createMockPrompt({ name: 'Blog Post', keywords: ['writing', 'content'] })
    ];

    beforeEach(() => {
      apiClient.getPrompts.mockResolvedValue({ results: mockPrompts });
      apiClient.searchPrompts.mockImplementation((query) => {
        const filtered = mockPrompts.filter(p => 
          p.name.toLowerCase().includes(query.toLowerCase()) ||
          p.keywords.some(k => k.includes(query.toLowerCase()))
        );
        return Promise.resolve({ results: filtered });
      });
      apiClient.createPrompt.mockResolvedValue({ 
        id: '4', 
        ...createMockPrompt({ name: 'New Prompt' }) 
      });
    });

    test('should create new prompt successfully', async () => {
      // Switch to prompts tab
      document.querySelector('[data-tab="prompts"]').click();
      await waitForAsync();
      
      // Click create button
      document.getElementById('createPrompt').click();
      
      // Modal should open
      const modal = document.getElementById('prompt-modal');
      modal.style.display = 'block';
      
      // Fill form
      simulateUserInput(document.getElementById('prompt-name'), 'Meeting Summary');
      simulateUserInput(document.getElementById('prompt-content'), 'Summarize the key points from this meeting transcript');
      simulateUserInput(document.getElementById('prompt-keywords'), 'meeting, summary, business');
      
      // Save
      document.getElementById('save-prompt').click();
      
      await waitForAsync();
      
      // Verify API call
      expect(apiClient.createPrompt).toHaveBeenCalledWith({
        name: 'Meeting Summary',
        content: 'Summarize the key points from this meeting transcript',
        keywords: ['meeting', 'summary', 'business']
      });
      
      // Modal should close
      expect(modal.style.display).toBe('none');
    });

    test('should search and filter prompts', async () => {
      // Load initial prompts
      await apiClient.getPrompts();
      
      // Display them
      const promptsList = document.getElementById('promptsList');
      mockPrompts.forEach(prompt => {
        const div = document.createElement('div');
        div.className = 'prompt-card';
        div.textContent = prompt.name;
        promptsList.appendChild(div);
      });
      
      // Search for "code"
      const searchInput = document.getElementById('promptSearch');
      simulateUserInput(searchInput, 'code');
      document.getElementById('searchPromptsBtn').click();
      
      await waitForAsync();
      
      // Verify search API call
      expect(apiClient.searchPrompts).toHaveBeenCalledWith('code');
      
      // Should filter results
      expect(promptsList.children.length).toBe(1);
      expect(promptsList.textContent).toContain('Code Review');
    });
  });

  describe('Media Processing Workflow', () => {
    const mockMediaItems = [
      createMockMediaItem({ title: 'Tutorial Video', type: 'video' }),
      createMockMediaItem({ title: 'Podcast Episode', type: 'audio' })
    ];

    beforeEach(() => {
      apiClient.getMediaList.mockResolvedValue({ results: mockMediaItems });
      apiClient.processUrl.mockResolvedValue({ 
        id: '3',
        status: 'processing',
        message: 'Processing started'
      });
    });

    test('should process URL and show in media list', async () => {
      // Switch to media tab
      document.querySelector('[data-tab="media"]').click();
      await waitForAsync();
      
      // Enter URL
      const urlInput = document.getElementById('mediaUrl');
      simulateUserInput(urlInput, 'https://youtube.com/watch?v=test123');
      
      // Process URL
      document.getElementById('processUrl').click();
      
      await waitForAsync();
      
      // Verify API call
      expect(apiClient.processUrl).toHaveBeenCalledWith(
        'https://youtube.com/watch?v=test123',
        expect.any(String)
      );
      
      // Load updated media list
      await apiClient.getMediaList();
      
      // Display media items
      const mediaList = document.getElementById('mediaList');
      mockMediaItems.forEach(item => {
        const div = document.createElement('div');
        div.className = 'media-item';
        div.innerHTML = `
          <h4>${item.title}</h4>
          <span class="media-type">${item.type}</span>
        `;
        mediaList.appendChild(div);
      });
      
      // Verify media displayed
      expect(mediaList.children.length).toBe(2);
      expect(mediaList.textContent).toContain('Tutorial Video');
      expect(mediaList.textContent).toContain('Podcast Episode');
    });
  });

  describe('Export/Import Workflow', () => {
    test('should export prompts successfully', async () => {
      const mockExportData = {
        filename: 'prompts_export_2024.md',
        content: btoa('# My Prompts\n\n## Email Writer\n...')
      };
      
      apiClient.exportPrompts.mockResolvedValue(mockExportData);
      
      // Mock chrome downloads API
      chrome.downloads = {
        download: jest.fn().mockResolvedValue(1)
      };
      
      // Switch to prompts tab
      document.querySelector('[data-tab="prompts"]').click();
      
      // Create export button (would normally be in the UI)
      const exportBtn = document.createElement('button');
      exportBtn.id = 'exportPrompts';
      exportBtn.textContent = 'Export';
      document.getElementById('prompts-tab').appendChild(exportBtn);
      
      // Click export
      exportBtn.click();
      
      await waitForAsync();
      
      // Verify export API call
      expect(apiClient.exportPrompts).toHaveBeenCalledWith('markdown');
      
      // Verify download initiated
      expect(chrome.downloads.download).toHaveBeenCalledWith(
        expect.objectContaining({
          filename: mockExportData.filename,
          saveAs: true
        })
      );
    });
  });

  describe('Error Handling in Workflows', () => {
    test('should handle API errors gracefully', async () => {
      // Simulate API error
      apiClient.createChatCompletion.mockRejectedValue(
        new Error('API rate limit exceeded')
      );
      
      // Try to send message
      const chatInput = document.getElementById('chatInput');
      simulateUserInput(chatInput, 'Hello');
      document.getElementById('sendMessage').click();
      
      await waitForAsync();
      
      // Should show error toast
      expect(document).toHaveToastNotification('error', /rate limit/i);
      
      // Input should not be cleared
      expect(chatInput.value).toBe('Hello');
    });

    test('should handle network failures', async () => {
      // Simulate network error
      apiClient.checkConnection.mockRejectedValue(
        new Error('Network error')
      );
      
      // Should show connection error
      await waitForAsync();
      
      expect(document.getElementById('connectionText').textContent).toContain('Error');
    });
  });

  describe('Tab Navigation Workflow', () => {
    test('should switch between tabs correctly', async () => {
      const tabs = ['chat', 'prompts', 'characters', 'media'];
      
      for (const tabName of tabs) {
        // Click tab button
        const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
        tabButton.click();
        
        await waitForAsync();
        
        // Verify tab is active
        expect(tabButton.classList.contains('active')).toBe(true);
        
        // Verify content pane is visible
        const tabPane = document.getElementById(`${tabName}-tab`);
        expect(tabPane.classList.contains('active')).toBe(true);
        
        // Other tabs should be hidden
        tabs.filter(t => t !== tabName).forEach(otherTab => {
          const otherPane = document.getElementById(`${otherTab}-tab`);
          expect(otherPane.classList.contains('active')).toBe(false);
        });
      }
    });
  });

  describe('Settings Persistence Workflow', () => {
    test('should persist user preferences across sessions', async () => {
      // Change model selection
      const modelSelect = document.getElementById('modelSelect');
      modelSelect.value = 'gpt-4';
      modelSelect.dispatchEvent(new Event('change'));
      
      // Verify saved to storage
      await waitForAsync();
      expect(chrome.storage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          selectedModel: 'gpt-4'
        })
      );
      
      // Simulate popup reload
      document.body.innerHTML = '';
      jest.resetModules();
      
      // Mock storage to return saved value
      chrome.storage.sync.get.mockImplementation((keys, callback) => {
        callback({ selectedModel: 'gpt-4' });
      });
      
      // Reload popup
      require('../../js/popup.js');
      await waitForAsync();
      
      // Model should be pre-selected
      expect(document.getElementById('modelSelect').value).toBe('gpt-4');
    });
  });
});