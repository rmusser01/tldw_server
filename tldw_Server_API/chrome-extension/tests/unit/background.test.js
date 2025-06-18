describe('Background Service Worker', () => {
  let backgroundScript;
  let mockFetch;

  beforeEach(() => {
    // Clear all chrome API mocks
    jest.clearAllMocks();
    
    // Setup fetch mock
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    
    // Reset chrome.runtime.onInstalled listeners
    chrome.runtime.onInstalled.addListener.mockClear();
    chrome.contextMenus.onClicked.addListener.mockClear();
    chrome.commands.onCommand.addListener.mockClear();
    chrome.runtime.onMessage.addListener.mockClear();
  });

  afterEach(() => {
    // Clear any intervals
    jest.clearAllTimers();
  });

  describe('Installation and setup', () => {
    it('should create context menus on installation', async () => {
      await import('../../js/background.js');
      
      // Get the onInstalled callback
      const onInstalledCallback = chrome.runtime.onInstalled.addListener.mock.calls[0][0];
      
      // Trigger installation
      onInstalledCallback({ reason: 'install' });

      expect(chrome.contextMenus.create).toHaveBeenCalledWith({
        id: 'send-to-chat',
        title: 'Send to TLDW Chat',
        contexts: ['selection']
      });

      expect(chrome.contextMenus.create).toHaveBeenCalledWith({
        id: 'process-as-media',
        title: 'Process as Media',
        contexts: ['selection', 'link', 'image', 'video', 'audio']
      });

      expect(chrome.contextMenus.create).toHaveBeenCalledWith({
        id: 'save-as-prompt',
        title: 'Save as Prompt',
        contexts: ['selection']
      });
    });

    it('should open options page on first install', async () => {
      await import('../../js/background.js');
      
      const onInstalledCallback = chrome.runtime.onInstalled.addListener.mock.calls[0][0];
      onInstalledCallback({ reason: 'install' });

      expect(chrome.runtime.openOptionsPage).toHaveBeenCalled();
    });
  });

  describe('Context menu handling', () => {
    beforeEach(async () => {
      await import('../../js/background.js');
    });

    it('should handle send-to-chat context menu click', async () => {
      const onClickedCallback = chrome.contextMenus.onClicked.addListener.mock.calls[0][0];
      
      const info = {
        menuItemId: 'send-to-chat',
        selectionText: 'Selected text'
      };
      const tab = {
        id: 1,
        url: 'https://example.com',
        title: 'Example Page'
      };

      chrome.storage.local.set = jest.fn().mockResolvedValue();
      chrome.action.openPopup = jest.fn();

      await onClickedCallback(info, tab);

      expect(chrome.storage.local.set).toHaveBeenCalledWith({
        pendingChatText: 'Selected text',
        sourceUrl: 'https://example.com',
        sourceTitle: 'Example Page'
      });
      expect(chrome.action.openPopup).toHaveBeenCalled();
    });

    it('should handle process-as-media with link URL', async () => {
      const onClickedCallback = chrome.contextMenus.onClicked.addListener.mock.calls[0][0];
      
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true })
      });

      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      const info = {
        menuItemId: 'process-as-media',
        linkUrl: 'https://example.com/video.mp4',
        selectionText: 'Some text'
      };
      const tab = {
        id: 1,
        title: 'Example Page'
      };

      await onClickedCallback(info, tab);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/media/ingest-web-content',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Token': 'Bearer test-token'
          },
          body: JSON.stringify({
            url: 'https://example.com/video.mp4',
            title: 'Example Page',
            media_type: 'link',
            selected_text: 'Some text'
          })
        })
      );

      expect(chrome.notifications.create).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Success',
          message: 'Content processed successfully'
        })
      );
    });

    it('should handle save-as-prompt context menu click', async () => {
      const onClickedCallback = chrome.contextMenus.onClicked.addListener.mock.calls[0][0];
      
      const info = {
        menuItemId: 'save-as-prompt',
        selectionText: 'Prompt text'
      };
      const tab = {
        id: 1,
        url: 'https://example.com',
        title: 'Example Page'
      };

      chrome.storage.local.set = jest.fn().mockResolvedValue();

      await onClickedCallback(info, tab);

      expect(chrome.storage.local.set).toHaveBeenCalledWith({
        pendingPromptText: 'Prompt text',
        sourceUrl: 'https://example.com',
        sourceTitle: 'Example Page'
      });
      expect(chrome.runtime.openOptionsPage).toHaveBeenCalled();
    });
  });

  describe('Keyboard shortcuts', () => {
    beforeEach(async () => {
      await import('../../js/background.js');
    });

    it('should handle send-to-chat keyboard shortcut', async () => {
      const onCommandCallback = chrome.commands.onCommand.addListener.mock.calls[0][0];
      
      chrome.tabs.query.mockResolvedValue([{ id: 1 }]);
      chrome.tabs.sendMessage.mockImplementation((tabId, message, callback) => {
        callback({ text: 'Selected text from page' });
      });
      chrome.storage.local.set = jest.fn().mockResolvedValue();

      await onCommandCallback('send-to-chat');

      expect(chrome.tabs.query).toHaveBeenCalledWith({
        active: true,
        currentWindow: true
      });
      expect(chrome.tabs.sendMessage).toHaveBeenCalledWith(
        1,
        { action: 'getSelection' },
        expect.any(Function)
      );
    });
  });

  describe('Message handling', () => {
    beforeEach(async () => {
      await import('../../js/background.js');
    });

    it('should handle processSelection message', () => {
      const onMessageCallback = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      
      const request = {
        action: 'processSelection',
        data: {
          type: 'send-to-chat',
          text: 'Selected text',
          url: 'https://example.com',
          title: 'Page Title'
        }
      };
      const sender = { tab: { id: 1 } };
      const sendResponse = jest.fn();

      onMessageCallback(request, sender, sendResponse);

      expect(sendResponse).toHaveBeenCalledWith({ success: true });
    });

    it('should handle apiRequest message', async () => {
      const onMessageCallback = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'response' })
      });

      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      const request = {
        action: 'apiRequest',
        endpoint: '/test',
        options: { method: 'GET' }
      };
      const sendResponse = jest.fn();

      const result = onMessageCallback(request, {}, sendResponse);
      
      // Should return true to keep channel open
      expect(result).toBe(true);

      // Wait for async operation
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/test',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Token': 'Bearer test-token'
          })
        })
      );
    });
  });

  describe('Connection monitoring', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should periodically check connection status', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      await import('../../js/background.js');

      // Fast-forward time to trigger interval
      jest.advanceTimersByTime(30000);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/media/',
        expect.objectContaining({
          method: 'GET'
        })
      );

      expect(chrome.action.setBadgeText).toHaveBeenCalledWith({ text: '' });
    });

    it('should show error badge when connection fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));
      
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      await import('../../js/background.js');

      jest.advanceTimersByTime(30000);

      // Wait for async operation
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(chrome.action.setBadgeText).toHaveBeenCalledWith({ text: '!' });
      expect(chrome.action.setBadgeBackgroundColor).toHaveBeenCalledWith({
        color: '#e74c3c'
      });
    });
  });

  describe('Error handling', () => {
    beforeEach(async () => {
      await import('../../js/background.js');
    });

    it('should show error notification when media processing fails', async () => {
      const onClickedCallback = chrome.contextMenus.onClicked.addListener.mock.calls[0][0];
      
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500
      });

      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      const info = {
        menuItemId: 'process-as-media',
        linkUrl: 'https://example.com/video.mp4'
      };
      const tab = { id: 1, title: 'Test' };

      await onClickedCallback(info, tab);

      expect(chrome.notifications.create).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Error',
          message: 'Failed to process content'
        })
      );
    });

    it('should show connection error notification', async () => {
      const onClickedCallback = chrome.contextMenus.onClicked.addListener.mock.calls[0][0];
      
      mockFetch.mockRejectedValue(new Error('Network error'));

      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      const info = {
        menuItemId: 'process-as-media',
        pageUrl: 'https://example.com'
      };
      const tab = { id: 1, title: 'Test' };

      await onClickedCallback(info, tab);

      expect(chrome.notifications.create).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Error',
          message: 'Failed to connect to server'
        })
      );
    });
  });
});