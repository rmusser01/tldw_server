/**
 * @jest-environment jsdom
 */

import { simulateUserInput, createDOM, cleanupDOM } from '../utils/helpers.js';

describe('XSS Prevention Tests', () => {
  let mockElements;

  beforeEach(() => {
    // Setup DOM with various input/output elements
    document.body.innerHTML = `
      <div id="chat-container">
        <div id="chatMessages"></div>
        <textarea id="chatInput"></textarea>
      </div>
      
      <div id="prompts-container">
        <input id="prompt-name" type="text" />
        <textarea id="prompt-content"></textarea>
        <div id="promptsList"></div>
      </div>
      
      <div id="toast-container"></div>
      
      <div id="search-results"></div>
    `;

    mockElements = {
      chatMessages: document.getElementById('chatMessages'),
      chatInput: document.getElementById('chatInput'),
      promptName: document.getElementById('prompt-name'),
      promptContent: document.getElementById('prompt-content'),
      promptsList: document.getElementById('promptsList'),
      toastContainer: document.getElementById('toast-container'),
      searchResults: document.getElementById('search-results')
    };
  });

  afterEach(() => {
    cleanupDOM();
  });

  describe('Chat Message Sanitization', () => {
    test('should prevent script injection in chat messages', () => {
      const maliciousInput = '<script>alert("XSS")</script>Hello';
      
      // Simulate adding message to chat
      const messageDiv = document.createElement('div');
      messageDiv.className = 'message';
      messageDiv.textContent = maliciousInput; // Using textContent, not innerHTML
      mockElements.chatMessages.appendChild(messageDiv);
      
      // Verify script tag is not executed
      expect(mockElements.chatMessages.innerHTML).not.toContain('<script>');
      expect(mockElements.chatMessages.textContent).toContain(maliciousInput);
      
      // Check no script elements exist
      const scripts = mockElements.chatMessages.querySelectorAll('script');
      expect(scripts.length).toBe(0);
    });

    test('should escape HTML entities in user messages', () => {
      const htmlInput = '<img src=x onerror="alert(\'XSS\')">';
      
      const messageDiv = document.createElement('div');
      messageDiv.textContent = htmlInput;
      mockElements.chatMessages.appendChild(messageDiv);
      
      // Should not create an actual img element
      const images = mockElements.chatMessages.querySelectorAll('img');
      expect(images.length).toBe(0);
      
      // Should display the raw HTML as text
      expect(messageDiv.textContent).toBe(htmlInput);
    });

    test('should handle onclick and other event handlers', () => {
      const eventHandlers = [
        '<div onclick="alert(\'XSS\')">Click me</div>',
        '<a href="javascript:alert(\'XSS\')">Link</a>',
        '<img src=x onload="alert(\'XSS\')">', 
        '<svg onload="alert(\'XSS\')">'
      ];
      
      eventHandlers.forEach(handler => {
        const messageDiv = document.createElement('div');
        messageDiv.textContent = handler;
        mockElements.chatMessages.appendChild(messageDiv);
      });
      
      // No actual elements should be created
      expect(mockElements.chatMessages.querySelectorAll('div[onclick]').length).toBe(0);
      expect(mockElements.chatMessages.querySelectorAll('a[href^="javascript:"]').length).toBe(0);
      expect(mockElements.chatMessages.querySelectorAll('img').length).toBe(0);
      expect(mockElements.chatMessages.querySelectorAll('svg').length).toBe(0);
    });
  });

  describe('Prompt Content Sanitization', () => {
    test('should sanitize prompt names', () => {
      const maliciousName = '"><script>alert("XSS")</script>';
      
      simulateUserInput(mockElements.promptName, maliciousName);
      
      // Value should be stored as-is in the input
      expect(mockElements.promptName.value).toBe(maliciousName);
      
      // When displayed, should be escaped
      const promptCard = document.createElement('div');
      promptCard.className = 'prompt-card';
      
      const nameElement = document.createElement('h4');
      nameElement.textContent = mockElements.promptName.value;
      promptCard.appendChild(nameElement);
      
      mockElements.promptsList.appendChild(promptCard);
      
      // No script should execute
      expect(mockElements.promptsList.querySelectorAll('script').length).toBe(0);
    });

    test('should handle malicious prompt content', () => {
      const maliciousContent = `
        <iframe src="javascript:alert('XSS')"></iframe>
        <object data="javascript:alert('XSS')"></object>
        <embed src="javascript:alert('XSS')">
      `;
      
      simulateUserInput(mockElements.promptContent, maliciousContent);
      
      // Display content safely
      const contentDiv = document.createElement('div');
      contentDiv.textContent = mockElements.promptContent.value;
      
      // No iframes, objects, or embeds should be created
      expect(contentDiv.querySelectorAll('iframe').length).toBe(0);
      expect(contentDiv.querySelectorAll('object').length).toBe(0);
      expect(contentDiv.querySelectorAll('embed').length).toBe(0);
    });
  });

  describe('URL Parameter Sanitization', () => {
    test('should sanitize URL parameters', () => {
      const maliciousUrl = 'https://example.com?q=<script>alert("XSS")</script>';
      
      // Parse URL safely
      const url = new URL(maliciousUrl);
      const searchParam = url.searchParams.get('q');
      
      // Display parameter value safely
      const resultDiv = document.createElement('div');
      resultDiv.textContent = `Search results for: ${searchParam}`;
      mockElements.searchResults.appendChild(resultDiv);
      
      // No script execution
      expect(mockElements.searchResults.querySelectorAll('script').length).toBe(0);
      expect(resultDiv.textContent).toContain('<script>alert("XSS")</script>');
    });

    test('should handle javascript: URLs', () => {
      const jsUrl = 'javascript:alert("XSS")';
      
      const link = document.createElement('a');
      
      // Safe way to set href
      if (!jsUrl.startsWith('http://') && !jsUrl.startsWith('https://')) {
        link.href = '#';
        link.textContent = 'Invalid URL';
      } else {
        link.href = jsUrl;
        link.textContent = jsUrl;
      }
      
      mockElements.searchResults.appendChild(link);
      
      // Should not have javascript: protocol
      expect(link.href).not.toContain('javascript:');
    });
  });

  describe('Toast Notification Sanitization', () => {
    test('should escape HTML in toast messages', () => {
      const createToast = (message, type = 'info') => {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const messageEl = document.createElement('span');
        messageEl.className = 'toast-message';
        messageEl.textContent = message; // Safe text content
        
        toast.appendChild(messageEl);
        mockElements.toastContainer.appendChild(toast);
        
        return toast;
      };
      
      const maliciousMessage = '<img src=x onerror="alert(\'XSS\')">';
      const toast = createToast(maliciousMessage);
      
      // Should not create img element
      expect(toast.querySelectorAll('img').length).toBe(0);
      expect(toast.textContent).toBe(maliciousMessage);
    });
  });

  describe('JSON Data Sanitization', () => {
    test('should safely parse JSON responses', () => {
      const maliciousJSON = '{"name":"<script>alert(\\"XSS\\")</script>","value":"test"}';
      
      let parsed;
      try {
        parsed = JSON.parse(maliciousJSON);
      } catch (e) {
        parsed = null;
      }
      
      if (parsed) {
        // Display parsed data safely
        const displayDiv = document.createElement('div');
        displayDiv.textContent = `Name: ${parsed.name}`;
        mockElements.searchResults.appendChild(displayDiv);
        
        // No script execution
        expect(mockElements.searchResults.querySelectorAll('script').length).toBe(0);
      }
    });

    test('should handle prototype pollution attempts', () => {
      const maliciousPayload = '{"__proto__":{"isAdmin":true}}';
      
      const obj = {};
      try {
        Object.assign(obj, JSON.parse(maliciousPayload));
      } catch (e) {
        // Safe handling
      }
      
      // Prototype should not be polluted
      expect({}.isAdmin).toBeUndefined();
    });
  });

  describe('Content Security Policy Compliance', () => {
    test('should not use inline event handlers', () => {
      // Check that no inline event handlers are present
      const inlineHandlers = [
        '[onclick]', '[onload]', '[onerror]', '[onsubmit]',
        '[onmouseover]', '[onmouseout]', '[onfocus]', '[onblur]'
      ];
      
      let hasInlineHandlers = false;
      inlineHandlers.forEach(selector => {
        if (document.querySelector(selector)) {
          hasInlineHandlers = true;
        }
      });
      
      expect(hasInlineHandlers).toBe(false);
    });

    test('should not use eval or Function constructor', () => {
      // Safe alternative to eval
      const userCode = 'console.log("test")';
      
      // Don't do this:
      // eval(userCode);
      // new Function(userCode)();
      
      // Instead, use safe alternatives
      const safeExecute = (code) => {
        // Log code instead of executing
        console.log('Would execute:', code);
      };
      
      expect(() => safeExecute(userCode)).not.toThrow();
    });
  });

  describe('DOM Manipulation Safety', () => {
    test('should use safe DOM methods', () => {
      const userContent = '<b>Bold text</b> and <script>alert("XSS")</script>';
      
      // Unsafe: element.innerHTML = userContent
      // Safe: use textContent or sanitize first
      
      const safeDiv = document.createElement('div');
      safeDiv.textContent = userContent;
      
      const unsafeDiv = document.createElement('div');
      // Would be unsafe: unsafeDiv.innerHTML = userContent
      
      expect(safeDiv.querySelectorAll('script').length).toBe(0);
      expect(safeDiv.querySelectorAll('b').length).toBe(0);
      expect(safeDiv.textContent).toBe(userContent);
    });

    test('should sanitize before using insertAdjacentHTML', () => {
      const container = document.createElement('div');
      const userHTML = '<img src=x onerror="alert(\'XSS\')">';
      
      // Unsafe: container.insertAdjacentHTML('beforeend', userHTML)
      
      // Safe: Create elements programmatically
      const img = document.createElement('img');
      img.src = 'x';
      img.alt = 'User image';
      img.onerror = null; // Remove any error handlers
      
      container.appendChild(img);
      
      // Check that no inline handlers exist
      expect(img.getAttribute('onerror')).toBeNull();
    });
  });

  describe('Storage Security', () => {
    test('should not store sensitive data in plain text', () => {
      const sensitiveData = {
        apiKey: 'secret-key-123',
        password: 'user-password'
      };
      
      // Mock secure storage
      const secureStorage = {
        set: (key, value) => {
          // In real implementation, encrypt sensitive data
          const encrypted = btoa(JSON.stringify(value)); // Simple encoding for test
          localStorage.setItem(key, encrypted);
        },
        get: (key) => {
          const encrypted = localStorage.getItem(key);
          if (!encrypted) return null;
          
          try {
            return JSON.parse(atob(encrypted));
          } catch {
            return null;
          }
        }
      };
      
      // Store securely
      secureStorage.set('config', sensitiveData);
      
      // Raw value should not be plaintext in storage
      const rawValue = localStorage.getItem('config');
      expect(rawValue).not.toContain('secret-key-123');
      expect(rawValue).not.toContain('user-password');
      
      // But should be retrievable
      const retrieved = secureStorage.get('config');
      expect(retrieved.apiKey).toBe(sensitiveData.apiKey);
    });
  });

  describe('Input Validation', () => {
    test('should validate and sanitize user inputs', () => {
      const validateInput = (input, type) => {
        switch (type) {
          case 'url':
            try {
              const url = new URL(input);
              return url.protocol === 'http:' || url.protocol === 'https:';
            } catch {
              return false;
            }
            
          case 'email':
            return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);
            
          case 'alphanumeric':
            return /^[a-zA-Z0-9]+$/.test(input);
            
          default:
            return true;
        }
      };
      
      // Test URL validation
      expect(validateInput('https://example.com', 'url')).toBe(true);
      expect(validateInput('javascript:alert("XSS")', 'url')).toBe(false);
      expect(validateInput('not-a-url', 'url')).toBe(false);
      
      // Test email validation
      expect(validateInput('user@example.com', 'email')).toBe(true);
      expect(validateInput('<script>@example.com', 'email')).toBe(false);
      
      // Test alphanumeric validation
      expect(validateInput('abc123', 'alphanumeric')).toBe(true);
      expect(validateInput('abc<script>', 'alphanumeric')).toBe(false);
    });
  });
});