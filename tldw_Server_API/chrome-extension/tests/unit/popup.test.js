/**
 * @jest-environment jsdom
 */

// Mock the API client
jest.mock('../../js/utils/api.js', () => ({
  apiClient: {
    checkConnection: jest.fn(),
    createChatCompletion: jest.fn(),
    getPrompts: jest.fn(),
    searchPrompts: jest.fn(),
    exportPrompts: jest.fn(),
    getCharacters: jest.fn(),
    searchCharacters: jest.fn(),
    importCharacterCard: jest.fn(),
    getMediaList: jest.fn(),
    getMediaItem: jest.fn(),
    processMediaUrl: jest.fn(),
    processMediaFile: jest.fn()
  }
}));

describe('Popup functionality', () => {
  let popup;
  const { apiClient } = require('../../js/utils/api.js');

  beforeEach(() => {
    // Setup DOM
    document.body.innerHTML = `
      <div id="connectionStatus" class="status-dot"></div>
      <div id="connectionText">Checking...</div>
      
      <div class="tab-button" data-tab="chat">Chat</div>
      <div class="tab-button" data-tab="prompts">Prompts</div>
      
      <div id="chat-tab" class="tab-pane active"></div>
      <div id="prompts-tab" class="tab-pane"></div>
      
      <select id="modelSelect">
        <option value="">Select Model...</option>
        <option value="gpt-4">GPT-4</option>
      </select>
      <select id="characterSelect">
        <option value="">No Character</option>
      </select>
      
      <div id="chatMessages"></div>
      <textarea id="chatInput"></textarea>
      <button id="sendMessage">Send</button>
      <button id="clearChat">Clear</button>
      
      <input id="promptSearch" />
      <button id="searchPromptsBtn">Search</button>
      <div id="promptsList"></div>
      <button id="exportPrompts">Export</button>
      
      <template id="message-template">
        <div class="message">
          <div class="message-role"></div>
          <div class="message-content"></div>
        </div>
      </template>
      
      <template id="prompt-item-template">
        <div class="prompt-item">
          <h4 class="prompt-name"></h4>
          <p class="prompt-details"></p>
          <div class="prompt-keywords"></div>
          <div class="prompt-actions">
            <button class="btn btn-small use-prompt">Use</button>
            <button class="btn btn-small edit-prompt">Edit</button>
          </div>
        </div>
      </template>
    `;

    // Clear all mocks
    jest.clearAllMocks();
  });

  describe('Connection status', () => {
    it('should show connected status when API is available', async () => {
      apiClient.checkConnection.mockResolvedValue(true);
      
      // Import popup.js after mocks are set up
      await import('../../js/popup.js');
      
      // Wait for async initialization
      await new Promise(resolve => setTimeout(resolve, 0));

      const statusDot = document.getElementById('connectionStatus');
      const statusText = document.getElementById('connectionText');

      expect(statusDot.classList.contains('connected')).toBe(true);
      expect(statusText.textContent).toBe('Connected');
    });

    it('should show disconnected status when API is unavailable', async () => {
      apiClient.checkConnection.mockResolvedValue(false);
      
      await import('../../js/popup.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      const statusDot = document.getElementById('connectionStatus');
      const statusText = document.getElementById('connectionText');

      expect(statusDot.classList.contains('error')).toBe(true);
      expect(statusText.textContent).toBe('Disconnected');
    });
  });

  describe('Tab functionality', () => {
    beforeEach(async () => {
      apiClient.checkConnection.mockResolvedValue(true);
      await import('../../js/popup.js');
    });

    it('should switch tabs when clicked', () => {
      const chatTab = document.querySelector('[data-tab="chat"]');
      const promptsTab = document.querySelector('[data-tab="prompts"]');
      const chatPane = document.getElementById('chat-tab');
      const promptsPane = document.getElementById('prompts-tab');

      // Click prompts tab
      promptsTab.click();

      expect(promptsTab.classList.contains('active')).toBe(true);
      expect(chatTab.classList.contains('active')).toBe(false);
      expect(promptsPane.classList.contains('active')).toBe(true);
      expect(chatPane.classList.contains('active')).toBe(false);
    });
  });

  describe('Chat functionality', () => {
    beforeEach(async () => {
      apiClient.checkConnection.mockResolvedValue(true);
      apiClient.createChatCompletion.mockResolvedValue({
        choices: [{
          message: { content: 'Hello! How can I help you?' }
        }]
      });
      await import('../../js/popup.js');
    });

    it('should send chat message when button clicked', async () => {
      const modelSelect = document.getElementById('modelSelect');
      const chatInput = document.getElementById('chatInput');
      const sendButton = document.getElementById('sendMessage');

      modelSelect.value = 'gpt-4';
      chatInput.value = 'Hello AI!';
      sendButton.click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(apiClient.createChatCompletion).toHaveBeenCalledWith({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello AI!' }],
        stream: false
      });

      const messages = document.querySelectorAll('.message');
      expect(messages.length).toBe(2); // User + Assistant message
    });

    it('should clear chat when clear button clicked', () => {
      // Add some messages first
      document.getElementById('chatMessages').innerHTML = '<div class="message">Test</div>';
      
      const clearButton = document.getElementById('clearChat');
      clearButton.click();

      expect(document.getElementById('chatMessages').innerHTML).toBe('');
    });

    it('should alert when no model selected', () => {
      window.alert = jest.fn();
      const chatInput = document.getElementById('chatInput');
      const sendButton = document.getElementById('sendMessage');

      chatInput.value = 'Hello AI!';
      sendButton.click();

      expect(window.alert).toHaveBeenCalledWith('Please enter a message and select a model');
    });
  });

  describe('Prompts functionality', () => {
    beforeEach(async () => {
      apiClient.checkConnection.mockResolvedValue(true);
      apiClient.getPrompts.mockResolvedValue({
        prompts: [
          {
            id: 1,
            name: 'Test Prompt',
            details: 'A test prompt',
            keywords: ['test', 'example']
          }
        ]
      });
      await import('../../js/popup.js');
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    it('should load prompts on initialization', () => {
      expect(apiClient.getPrompts).toHaveBeenCalledWith(1, 20);
      
      const promptItems = document.querySelectorAll('.prompt-item');
      expect(promptItems.length).toBe(1);
      expect(promptItems[0].querySelector('.prompt-name').textContent).toBe('Test Prompt');
    });

    it('should search prompts when search button clicked', async () => {
      apiClient.searchPrompts.mockResolvedValue({
        results: [
          {
            id: 2,
            name: 'Search Result',
            details: 'Found prompt'
          }
        ]
      });

      const searchInput = document.getElementById('promptSearch');
      const searchButton = document.getElementById('searchPromptsBtn');

      searchInput.value = 'test query';
      searchButton.click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(apiClient.searchPrompts).toHaveBeenCalledWith('test query');
    });

    it('should export prompts when export button clicked', async () => {
      apiClient.exportPrompts.mockResolvedValue({
        content: 'YmFzZTY0Y29udGVudA==',
        filename: 'prompts.md'
      });

      chrome.downloads.download = jest.fn();
      
      const exportButton = document.getElementById('exportPrompts');
      exportButton.click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(apiClient.exportPrompts).toHaveBeenCalledWith('markdown');
      expect(chrome.downloads.download).toHaveBeenCalled();
    });
  });

  describe('Character selection', () => {
    beforeEach(async () => {
      apiClient.checkConnection.mockResolvedValue(true);
      apiClient.getCharacters.mockResolvedValue([
        { id: 1, name: 'Character 1' },
        { id: 2, name: 'Character 2' }
      ]);
      await import('../../js/popup.js');
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    it('should populate character select on load', () => {
      const characterSelect = document.getElementById('characterSelect');
      expect(characterSelect.options.length).toBe(3); // No Character + 2 characters
      expect(characterSelect.options[1].text).toBe('Character 1');
      expect(characterSelect.options[2].text).toBe('Character 2');
    });

    it('should update selected character when changed', () => {
      const characterSelect = document.getElementById('characterSelect');
      characterSelect.value = '1';
      characterSelect.dispatchEvent(new Event('change'));

      // Send a message to verify character_id is included
      const modelSelect = document.getElementById('modelSelect');
      const chatInput = document.getElementById('chatInput');
      const sendButton = document.getElementById('sendMessage');

      modelSelect.value = 'gpt-4';
      chatInput.value = 'Hello!';
      sendButton.click();

      expect(apiClient.createChatCompletion).toHaveBeenCalledWith(
        expect.objectContaining({
          character_id: '1'
        })
      );
    });
  });

  describe('Error handling', () => {
    beforeEach(async () => {
      apiClient.checkConnection.mockResolvedValue(true);
      await import('../../js/popup.js');
    });

    it('should display error message when chat fails', async () => {
      apiClient.createChatCompletion.mockRejectedValue(new Error('API Error'));

      const modelSelect = document.getElementById('modelSelect');
      const chatInput = document.getElementById('chatInput');
      const sendButton = document.getElementById('sendMessage');

      modelSelect.value = 'gpt-4';
      chatInput.value = 'Hello!';
      sendButton.click();

      await new Promise(resolve => setTimeout(resolve, 0));

      const messages = document.querySelectorAll('.message');
      const lastMessage = messages[messages.length - 1];
      expect(lastMessage.querySelector('.message-content').textContent).toContain('Error: API Error');
    });
  });
});