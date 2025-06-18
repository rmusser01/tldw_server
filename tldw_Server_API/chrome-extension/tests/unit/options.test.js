/**
 * @jest-environment jsdom
 */

describe('Options Page', () => {
  let mockFetch;

  beforeEach(() => {
    // Setup DOM
    document.body.innerHTML = `
      <input id="serverUrl" />
      <input id="apiToken" type="password" />
      <select id="defaultModel">
        <option value="">None</option>
        <option value="gpt-4">GPT-4</option>
      </select>
      <input id="defaultTemperature" type="range" min="0" max="2" step="0.1" />
      <span id="temperatureValue">0.7</span>
      <input id="maxTokens" type="number" />
      <input id="autoLoadChats" type="checkbox" />
      <input id="streamResponses" type="checkbox" />
      <input id="showNotifications" type="checkbox" />
      
      <button id="saveSettings">Save</button>
      <button id="cancelSettings">Cancel</button>
      <button id="testConnection">Test Connection</button>
      <span id="connectionStatus" class="status-message"></span>
      
      <button id="exportSettings">Export</button>
      <button id="importSettings">Import</button>
      <input id="importFile" type="file" style="display: none;" />
      <button id="clearCache">Clear Cache</button>
      <button id="resetSettings">Reset All</button>
      
      <div id="promptCreation" style="display: none;">
        <input id="promptName" />
        <textarea id="promptContent"></textarea>
        <button id="savePrompt">Save Prompt</button>
        <button id="cancelPrompt">Cancel</button>
      </div>
    `;

    // Setup mocks
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    
    // Mock window methods
    window.close = jest.fn();
    window.alert = jest.fn();
    window.confirm = jest.fn();
    
    // Clear chrome mocks
    jest.clearAllMocks();
  });

  describe('Loading settings', () => {
    it('should load saved settings on initialization', async () => {
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://custom-server:8000',
        apiToken: 'custom-token',
        defaultModel: 'gpt-4',
        defaultTemperature: 0.8,
        maxTokens: 2000,
        autoLoadChats: true,
        streamResponses: true,
        showNotifications: false
      });

      await import('../../js/options.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(document.getElementById('serverUrl').value).toBe('http://custom-server:8000');
      expect(document.getElementById('apiToken').value).toBe('custom-token');
      expect(document.getElementById('defaultModel').value).toBe('gpt-4');
      expect(document.getElementById('defaultTemperature').value).toBe('0.8');
      expect(document.getElementById('temperatureValue').textContent).toBe('0.8');
      expect(document.getElementById('maxTokens').value).toBe('2000');
      expect(document.getElementById('autoLoadChats').checked).toBe(true);
      expect(document.getElementById('streamResponses').checked).toBe(true);
      expect(document.getElementById('showNotifications').checked).toBe(false);
    });

    it('should use defaults when no settings saved', async () => {
      chrome.storage.sync.get.mockResolvedValue({});

      await import('../../js/options.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(document.getElementById('serverUrl').value).toBe('http://localhost:8000');
      expect(document.getElementById('apiToken').value).toBe('');
      expect(document.getElementById('defaultTemperature').value).toBe('0.7');
    });
  });

  describe('Saving settings', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({});
      await import('../../js/options.js');
    });

    it('should save settings when save button clicked', async () => {
      // Set form values
      document.getElementById('serverUrl').value = 'http://new-server:8000';
      document.getElementById('apiToken').value = 'new-token';
      document.getElementById('defaultModel').value = 'gpt-4';
      document.getElementById('defaultTemperature').value = '1.2';
      document.getElementById('maxTokens').value = '3000';
      document.getElementById('autoLoadChats').checked = true;

      chrome.storage.sync.set.mockResolvedValue();

      // Click save
      document.getElementById('saveSettings').click();

      expect(chrome.storage.sync.set).toHaveBeenCalledWith({
        serverUrl: 'http://new-server:8000',
        apiToken: 'new-token',
        defaultModel: 'gpt-4',
        defaultTemperature: 1.2,
        maxTokens: 3000,
        autoLoadChats: true,
        streamResponses: false,
        showNotifications: true
      });

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        action: 'settingsUpdated'
      });

      // Wait for timeout
      await new Promise(resolve => setTimeout(resolve, 1600));
      expect(window.close).toHaveBeenCalled();
    });

    it('should close window when cancel clicked', () => {
      document.getElementById('cancelSettings').click();
      expect(window.close).toHaveBeenCalled();
    });
  });

  describe('Connection testing', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({});
      await import('../../js/options.js');
    });

    it('should test connection successfully', async () => {
      document.getElementById('serverUrl').value = 'http://test-server:8000';
      document.getElementById('apiToken').value = 'test-token';

      mockFetch.mockResolvedValue({ ok: true });

      document.getElementById('testConnection').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-server:8000/api/v1/media/',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Token': 'Bearer test-token'
          }
        })
      );

      const status = document.getElementById('connectionStatus');
      expect(status.textContent).toBe('Connection successful!');
      expect(status.classList.contains('success')).toBe(true);
    });

    it('should show authentication error', async () => {
      document.getElementById('serverUrl').value = 'http://test-server:8000';
      document.getElementById('apiToken').value = 'wrong-token';

      mockFetch.mockResolvedValue({ ok: false, status: 401 });

      document.getElementById('testConnection').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      const status = document.getElementById('connectionStatus');
      expect(status.textContent).toBe('Authentication failed. Check your API token.');
      expect(status.classList.contains('error')).toBe(true);
    });

    it('should show connection error', async () => {
      document.getElementById('serverUrl').value = 'http://invalid-server';

      mockFetch.mockRejectedValue(new Error('Network error'));

      document.getElementById('testConnection').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      const status = document.getElementById('connectionStatus');
      expect(status.textContent).toBe('Connection failed. Check server URL and ensure server is running.');
      expect(status.classList.contains('error')).toBe(true);
    });
  });

  describe('Temperature slider', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({});
      await import('../../js/options.js');
    });

    it('should update temperature value display', () => {
      const temperatureSlider = document.getElementById('defaultTemperature');
      const temperatureValue = document.getElementById('temperatureValue');

      temperatureSlider.value = '1.5';
      temperatureSlider.dispatchEvent(new Event('input'));

      expect(temperatureValue.textContent).toBe('1.5');
    });
  });

  describe('Import/Export settings', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });
      await import('../../js/options.js');
    });

    it('should export settings as JSON', async () => {
      const createElementSpy = jest.spyOn(document, 'createElement');
      const mockAnchor = { href: '', download: '', click: jest.fn() };
      createElementSpy.mockReturnValue(mockAnchor);

      URL.createObjectURL = jest.fn(() => 'blob:mock-url');
      URL.revokeObjectURL = jest.fn();

      document.getElementById('exportSettings').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(chrome.storage.sync.get).toHaveBeenCalledWith(null);
      expect(mockAnchor.download).toBe('tldw-assistant-settings.json');
      expect(mockAnchor.click).toHaveBeenCalled();
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('should import settings from JSON file', async () => {
      const mockFile = new File(
        [JSON.stringify({ serverUrl: 'http://imported:8000', apiToken: 'imported-token' })],
        'settings.json',
        { type: 'application/json' }
      );

      const fileInput = document.getElementById('importFile');
      Object.defineProperty(fileInput, 'files', {
        value: [mockFile],
        writable: false
      });

      chrome.storage.sync.set.mockResolvedValue();

      document.getElementById('importSettings').click();
      fileInput.dispatchEvent(new Event('change'));

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(chrome.storage.sync.set).toHaveBeenCalledWith({
        serverUrl: 'http://imported:8000',
        apiToken: 'imported-token'
      });
    });
  });

  describe('Data management', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({});
      await import('../../js/options.js');
    });

    it('should clear cache when confirmed', async () => {
      window.confirm.mockReturnValue(true);
      chrome.storage.local.clear.mockResolvedValue();

      document.getElementById('clearCache').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(window.confirm).toHaveBeenCalledWith('Are you sure you want to clear all cached data?');
      expect(chrome.storage.local.clear).toHaveBeenCalled();

      const status = document.getElementById('connectionStatus');
      expect(status.textContent).toBe('Cache cleared successfully!');
    });

    it('should not clear cache when cancelled', () => {
      window.confirm.mockReturnValue(false);

      document.getElementById('clearCache').click();

      expect(chrome.storage.local.clear).not.toHaveBeenCalled();
    });

    it('should reset all settings when confirmed', async () => {
      window.confirm.mockReturnValue(true);
      chrome.storage.sync.clear.mockResolvedValue();
      chrome.storage.local.clear.mockResolvedValue();

      document.getElementById('resetSettings').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(window.confirm).toHaveBeenCalledWith('Are you sure you want to reset all settings to defaults?');
      expect(chrome.storage.sync.clear).toHaveBeenCalled();
      expect(chrome.storage.local.clear).toHaveBeenCalled();

      const status = document.getElementById('connectionStatus');
      expect(status.textContent).toBe('Settings reset to defaults!');
    });
  });

  describe('Prompt creation', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });
      chrome.storage.local.get.mockResolvedValue({
        pendingPromptText: 'This is pending prompt text'
      });
      await import('../../js/options.js');
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    it('should show prompt creation form when pending prompt exists', () => {
      const promptCreation = document.getElementById('promptCreation');
      const promptContent = document.getElementById('promptContent');

      expect(promptCreation.style.display).toBe('block');
      expect(promptContent.value).toBe('This is pending prompt text');
    });

    it('should save prompt successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1, name: 'New Prompt' })
      });

      document.getElementById('promptName').value = 'Test Prompt';
      document.getElementById('promptContent').value = 'Test content';

      document.getElementById('savePrompt').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/prompts/',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Token': 'Bearer test-token'
          },
          body: JSON.stringify({
            name: 'Test Prompt',
            system_prompt: 'Test content',
            details: 'Created from browser extension'
          })
        })
      );

      expect(window.alert).toHaveBeenCalledWith('Prompt saved successfully!');
      expect(chrome.storage.local.remove).toHaveBeenCalledWith(['pendingPromptText']);
    });

    it('should show error when prompt save fails', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        json: async () => ({ detail: 'Prompt already exists' })
      });

      document.getElementById('promptName').value = 'Duplicate';
      document.getElementById('promptContent').value = 'Content';

      document.getElementById('savePrompt').click();

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(window.alert).toHaveBeenCalledWith('Failed to save prompt: Prompt already exists');
    });

    it('should cancel prompt creation', () => {
      const promptCreation = document.getElementById('promptCreation');
      
      document.getElementById('cancelPrompt').click();

      expect(promptCreation.style.display).toBe('none');
      expect(chrome.storage.local.remove).toHaveBeenCalledWith(['pendingPromptText']);
    });
  });
});