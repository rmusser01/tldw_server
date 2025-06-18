/**
 * @jest-environment jsdom
 */

describe('Content Script', () => {
  let contentScript;

  beforeEach(() => {
    // Reset DOM
    document.body.innerHTML = `
      <div id="test-content">
        <p>Test paragraph with some text</p>
        <div>Another div with content</div>
      </div>
    `;

    // Clear chrome mocks
    jest.clearAllMocks();
    
    // Mock window.getSelection
    window.getSelection = jest.fn(() => ({
      toString: () => 'Selected text from page'
    }));

    // Setup chrome.storage mock
    chrome.storage.sync.get.mockResolvedValue({
      showFloatingButton: true
    });
  });

  describe('Message handling', () => {
    beforeEach(async () => {
      await import('../../js/content.js');
    });

    it('should respond to getSelection message', () => {
      const messageListener = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      const sendResponse = jest.fn();

      messageListener(
        { action: 'getSelection' },
        {},
        sendResponse
      );

      expect(sendResponse).toHaveBeenCalledWith({
        text: 'Selected text from page'
      });
    });

    it('should respond to getPageInfo message', () => {
      const messageListener = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      const sendResponse = jest.fn();

      document.title = 'Test Page Title';
      window.location.href = 'https://example.com/test';

      messageListener(
        { action: 'getPageInfo' },
        {},
        sendResponse
      );

      expect(sendResponse).toHaveBeenCalledWith({
        title: 'Test Page Title',
        url: 'https://example.com/test',
        selectedText: 'Selected text from page',
        pageContent: expect.stringContaining('Test paragraph')
      });
    });
  });

  describe('Floating button', () => {
    it('should create floating button when enabled', async () => {
      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: true
      });

      await import('../../js/content.js');
      
      // Wait for async initialization
      await new Promise(resolve => setTimeout(resolve, 0));

      const floatingButton = document.querySelector('.tldw-floating-button');
      expect(floatingButton).toBeTruthy();
      expect(floatingButton.style.display).toBe('none'); // Initially hidden
    });

    it('should not create floating button when disabled', async () => {
      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: false
      });

      await import('../../js/content.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      const floatingButton = document.querySelector('.tldw-floating-button');
      expect(floatingButton).toBeFalsy();
    });

    it('should show floating button on text selection', async () => {
      jest.useFakeTimers();
      
      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: true
      });

      await import('../../js/content.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      const floatingButton = document.querySelector('.tldw-floating-button');
      
      // Simulate text selection
      window.getSelection.mockReturnValue({
        toString: () => 'This is a long enough selection'
      });

      // Trigger mouseup event
      document.dispatchEvent(new MouseEvent('mouseup'));
      
      // Fast-forward timers
      jest.advanceTimersByTime(500);

      expect(floatingButton.style.display).toBe('flex');
      
      jest.useRealTimers();
    });

    it('should hide floating button when selection is too short', async () => {
      jest.useFakeTimers();
      
      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: true
      });

      await import('../../js/content.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      const floatingButton = document.querySelector('.tldw-floating-button');
      
      // Show button first
      floatingButton.style.display = 'flex';
      
      // Simulate short text selection
      window.getSelection.mockReturnValue({
        toString: () => 'Short'
      });

      document.dispatchEvent(new MouseEvent('mouseup'));
      jest.advanceTimersByTime(500);

      expect(floatingButton.style.display).toBe('none');
      
      jest.useRealTimers();
    });
  });

  describe('Quick actions menu', () => {
    beforeEach(async () => {
      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: true
      });
      await import('../../js/content.js');
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    it('should show quick actions when floating button is clicked', () => {
      const floatingButton = document.querySelector('.tldw-floating-button');
      
      // Simulate text selection
      window.getSelection.mockReturnValue({
        toString: () => 'Selected text for actions'
      });

      // Click floating button
      const clickEvent = new MouseEvent('click', {
        pageX: 100,
        pageY: 200
      });
      floatingButton.dispatchEvent(clickEvent);

      const quickActions = document.querySelector('.tldw-quick-actions');
      expect(quickActions).toBeTruthy();
      expect(quickActions.textContent).toContain('Send to Chat');
      expect(quickActions.textContent).toContain('Save as Prompt');
      expect(quickActions.textContent).toContain('Process as Media');
    });

    it('should send message when quick action is clicked', () => {
      const floatingButton = document.querySelector('.tldw-floating-button');
      
      window.getSelection.mockReturnValue({
        toString: () => 'Text to process'
      });

      floatingButton.dispatchEvent(new MouseEvent('click', {
        pageX: 100,
        pageY: 200
      }));

      const sendToChatButton = Array.from(
        document.querySelectorAll('.tldw-quick-actions div')
      ).find(el => el.textContent === 'Send to Chat');

      sendToChatButton.click();

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        action: 'processSelection',
        data: {
          type: 'sendToChat',
          text: 'Text to process',
          url: window.location.href,
          title: document.title
        }
      });
    });

    it('should remove quick actions menu when clicking outside', async () => {
      jest.useFakeTimers();
      
      const floatingButton = document.querySelector('.tldw-floating-button');
      
      window.getSelection.mockReturnValue({
        toString: () => 'Selected text'
      });

      floatingButton.click();
      
      const quickActions = document.querySelector('.tldw-quick-actions');
      expect(quickActions).toBeTruthy();

      // Click outside after timeout
      jest.advanceTimersByTime(100);
      document.body.click();

      expect(document.querySelector('.tldw-quick-actions')).toBeFalsy();
      
      jest.useRealTimers();
    });
  });

  describe('Keyboard shortcuts', () => {
    beforeEach(async () => {
      await import('../../js/content.js');
    });

    it('should handle Ctrl+Shift+T shortcut', () => {
      window.getSelection.mockReturnValue({
        toString: () => 'Shortcut selected text'
      });

      const keyEvent = new KeyboardEvent('keydown', {
        key: 'T',
        ctrlKey: true,
        shiftKey: true
      });

      document.dispatchEvent(keyEvent);

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        action: 'processSelection',
        data: {
          type: 'sendToChat',
          text: 'Shortcut selected text',
          url: window.location.href,
          title: document.title
        }
      });
    });

    it('should handle Cmd+Shift+T on Mac', () => {
      window.getSelection.mockReturnValue({
        toString: () => 'Mac shortcut text'
      });

      const keyEvent = new KeyboardEvent('keydown', {
        key: 'T',
        metaKey: true,
        shiftKey: true
      });

      document.dispatchEvent(keyEvent);

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        action: 'processSelection',
        data: {
          type: 'sendToChat',
          text: 'Mac shortcut text',
          url: window.location.href,
          title: document.title
        }
      });
    });
  });

  describe('Element selection mode', () => {
    beforeEach(async () => {
      await import('../../js/content.js');
    });

    it('should enable element highlighting', () => {
      const messageListener = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      const sendResponse = jest.fn();

      messageListener(
        { action: 'enableElementSelection' },
        {},
        sendResponse
      );

      expect(sendResponse).toHaveBeenCalledWith({ success: true });

      // Simulate mouseover
      const testElement = document.querySelector('#test-content p');
      const mouseoverEvent = new MouseEvent('mouseover', {
        target: testElement
      });
      testElement.dispatchEvent(mouseoverEvent);

      expect(testElement.style.outline).toBe('2px solid #3498db');
    });

    it('should disable element highlighting', () => {
      const messageListener = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      const sendResponse = jest.fn();

      // Enable first
      messageListener({ action: 'enableElementSelection' }, {}, jest.fn());

      // Highlight an element
      const testElement = document.querySelector('#test-content p');
      testElement.style.outline = '2px solid #3498db';

      // Disable
      messageListener(
        { action: 'disableElementSelection' },
        {},
        sendResponse
      );

      expect(sendResponse).toHaveBeenCalledWith({ success: true });
      expect(testElement.style.outline).toBe('');
    });

    it('should select element on click when in selection mode', () => {
      const messageListener = chrome.runtime.onMessage.addListener.mock.calls[0][0];
      
      // Enable selection mode
      messageListener({ action: 'enableElementSelection' }, {}, jest.fn());

      const testElement = document.querySelector('#test-content p');
      testElement.textContent = 'Element content to process';

      // Simulate click
      const clickEvent = new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      });
      
      // Manually trigger since we're testing the handler
      const mouseoverEvent = new MouseEvent('mouseover');
      testElement.dispatchEvent(mouseoverEvent);
      
      testElement.dispatchEvent(clickEvent);

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        action: 'processSelection',
        data: {
          type: 'processAsMedia',
          text: 'Element content to process',
          url: window.location.href,
          title: document.title
        }
      });
    });
  });

  describe('Initialization', () => {
    it('should initialize when DOM is already loaded', async () => {
      // Set readyState before importing
      Object.defineProperty(document, 'readyState', {
        value: 'complete',
        writable: true
      });

      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: true
      });

      await import('../../js/content.js');
      await new Promise(resolve => setTimeout(resolve, 0));

      const floatingButton = document.querySelector('.tldw-floating-button');
      expect(floatingButton).toBeTruthy();
    });

    it('should wait for DOMContentLoaded when loading', async () => {
      // Set readyState to loading
      Object.defineProperty(document, 'readyState', {
        value: 'loading',
        writable: true
      });

      let domContentLoadedCallback;
      document.addEventListener = jest.fn((event, callback) => {
        if (event === 'DOMContentLoaded') {
          domContentLoadedCallback = callback;
        }
      });

      chrome.storage.sync.get.mockResolvedValue({
        showFloatingButton: true
      });

      await import('../../js/content.js');

      // Floating button shouldn't exist yet
      expect(document.querySelector('.tldw-floating-button')).toBeFalsy();

      // Trigger DOMContentLoaded
      domContentLoadedCallback();
      await new Promise(resolve => setTimeout(resolve, 0));

      // Now it should exist
      expect(document.querySelector('.tldw-floating-button')).toBeTruthy();
    });
  });
});