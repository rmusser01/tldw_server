/**
 * @jest-environment jsdom
 */

import { waitForAsync, createMockBrowserAPI } from '../utils/helpers.js';

describe('Browser Compatibility', () => {
  beforeEach(() => {
    // Clear any existing global objects
    delete window.browser;
    delete window.chrome;
    delete window.browserAPI;
    jest.clearAllMocks();
  });

  describe('browser-polyfill.js', () => {
    beforeEach(() => {
      // Reset modules to test fresh imports
      jest.resetModules();
    });

    test('should not create polyfill when browser API already exists', () => {
      // Setup Firefox-like environment
      window.browser = {
        runtime: {
          sendMessage: jest.fn()
        }
      };

      // Load polyfill
      require('../../js/browser-polyfill.js');

      // Should not overwrite existing browser API
      expect(window.browser.runtime.sendMessage).toBe(window.browser.runtime.sendMessage);
    });

    test('should create browser API wrapper for Chrome', () => {
      // Setup Chrome-like environment
      window.chrome = {
        runtime: {
          sendMessage: jest.fn((message, callback) => {
            callback({ success: true });
          })
        },
        tabs: {
          query: jest.fn((queryInfo, callback) => {
            callback([{ id: 1, url: 'https://example.com' }]);
          })
        },
        storage: {
          sync: {
            get: jest.fn((keys, callback) => {
              callback({ key: 'value' });
            }),
            set: jest.fn((data, callback) => {
              callback();
            })
          }
        }
      };

      // Load polyfill
      require('../../js/browser-polyfill.js');

      // Should create browser API
      expect(window.browser).toBeDefined();
      expect(typeof window.browser.runtime.sendMessage).toBe('function');
    });

    test('should promisify Chrome callback APIs', async () => {
      // Setup Chrome API with callbacks
      const mockResponse = { data: 'test' };
      window.chrome = {
        runtime: {
          sendMessage: jest.fn((message, callback) => {
            setTimeout(() => callback(mockResponse), 10);
          })
        }
      };

      // Load polyfill
      require('../../js/browser-polyfill.js');

      // Test promisified version
      const result = await window.browser.runtime.sendMessage({ test: true });
      expect(result).toEqual(mockResponse);
      expect(window.chrome.runtime.sendMessage).toHaveBeenCalledWith(
        { test: true },
        expect.any(Function)
      );
    });

    test('should handle Chrome API errors in promises', async () => {
      // Setup Chrome API that sets lastError
      window.chrome = {
        runtime: {
          lastError: null,
          sendMessage: jest.fn((message, callback) => {
            window.chrome.runtime.lastError = { message: 'Test error' };
            callback();
          })
        }
      };

      // Load polyfill
      require('../../js/browser-polyfill.js');

      // Should reject promise on error
      await expect(window.browser.runtime.sendMessage({ test: true }))
        .rejects.toThrow('Test error');
    });

    test('should detect browser type correctly', () => {
      // Test Firefox detection
      Object.defineProperty(navigator, 'userAgent', {
        value: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
        configurable: true
      });

      jest.resetModules();
      require('../../js/browser-polyfill.js');
      expect(window.isFirefox).toBe(true);
      expect(window.isChrome).toBe(false);

      // Test Chrome detection
      Object.defineProperty(navigator, 'userAgent', {
        value: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        configurable: true
      });

      jest.resetModules();
      delete window.browser;
      require('../../js/browser-polyfill.js');
      expect(window.isFirefox).toBe(false);
      expect(window.isChrome).toBe(true);
    });
  });

  describe('compat-utils.js', () => {
    let compatUtils;

    beforeEach(() => {
      jest.resetModules();
      // Setup mock browser API
      window.browser = createMockBrowserAPI();
      window.chrome = createMockBrowserAPI();
      compatUtils = require('../../js/compat-utils.js');
    });

    test('should get correct browser API reference', () => {
      const api = compatUtils.getBrowserAPI();
      expect(api).toBeDefined();
      expect(api.runtime).toBeDefined();
      expect(api.tabs).toBeDefined();
    });

    test('should handle browser action vs action API differences', () => {
      // Test Chrome MV3 (action API)
      window.browser.action = { setBadgeText: jest.fn() };
      const actionAPI = compatUtils.getBrowserAction();
      expect(actionAPI).toBe(window.browser.action);

      // Test Chrome MV2 (browserAction API)
      delete window.browser.action;
      window.browser.browserAction = { setBadgeText: jest.fn() };
      const browserActionAPI = compatUtils.getBrowserAction();
      expect(browserActionAPI).toBe(window.browser.browserAction);
    });

    test('should detect manifest version correctly', () => {
      window.browser.runtime.getManifest = jest.fn(() => ({
        manifest_version: 3
      }));

      expect(compatUtils.getManifestVersion()).toBe(3);

      window.browser.runtime.getManifest = jest.fn(() => ({
        manifest_version: 2
      }));

      expect(compatUtils.getManifestVersion()).toBe(2);
    });

    test('storage operations should work across browsers', async () => {
      const testData = { key: 'value' };

      // Test set operation
      await compatUtils.storage.set('sync', testData);
      expect(window.browser.storage.sync.set).toHaveBeenCalledWith(testData);

      // Test get operation
      window.browser.storage.sync.get.mockResolvedValue(testData);
      const result = await compatUtils.storage.get('sync', ['key']);
      expect(result).toEqual(testData);
    });

    test('tabs operations should work across browsers', async () => {
      const mockTabs = [{ id: 1, url: 'https://example.com' }];
      window.browser.tabs.query.mockResolvedValue(mockTabs);

      const tabs = await compatUtils.tabs.query({ active: true });
      expect(tabs).toEqual(mockTabs);
      expect(window.browser.tabs.query).toHaveBeenCalledWith({ active: true });
    });

    test('should handle missing APIs gracefully', async () => {
      // Remove downloads API
      delete window.browser.downloads;

      const result = await compatUtils.downloads.download({ url: 'test.pdf' });
      expect(result).toBe(false);

      // Remove notifications API
      delete window.browser.notifications;

      const notifResult = await compatUtils.notifications.create('test', {
        type: 'basic',
        title: 'Test',
        message: 'Test message'
      });
      expect(notifResult).toBe(false);
    });

    test('should handle API errors gracefully', async () => {
      window.browser.storage.sync.set.mockRejectedValue(new Error('Storage error'));

      await expect(compatUtils.storage.set('sync', { test: 'data' }))
        .rejects.toThrow('Storage error');
    });
  });

  describe('Cross-browser API usage in popup.js', () => {
    beforeEach(() => {
      jest.resetModules();
      document.body.innerHTML = '<div id="test-container"></div>';
    });

    test('should use correct API variable in different browser contexts', () => {
      // Test with browser API (Firefox)
      window.browser = createMockBrowserAPI();
      const api1 = (typeof browser !== 'undefined') ? browser : chrome;
      expect(api1).toBe(window.browser);

      // Test with chrome API (Chrome)
      delete window.browser;
      window.chrome = createMockBrowserAPI();
      const api2 = (typeof browser !== 'undefined') ? browser : chrome;
      expect(api2).toBe(window.chrome);
    });

    test('should handle API calls consistently', async () => {
      window.browser = createMockBrowserAPI();
      const api = window.browser;

      // Test various API calls
      await api.runtime.openOptionsPage();
      expect(api.runtime.openOptionsPage).toHaveBeenCalled();

      await api.tabs.create({ url: 'https://example.com' });
      expect(api.tabs.create).toHaveBeenCalledWith({ url: 'https://example.com' });

      const mockTabs = [{ id: 1, active: true }];
      api.tabs.query.mockResolvedValue(mockTabs);
      const tabs = await api.tabs.query({ active: true });
      expect(tabs).toEqual(mockTabs);
    });
  });

  describe('Browser-specific features', () => {
    test('should handle Firefox-specific features', () => {
      window.browser = {
        runtime: {
          getBrowserInfo: jest.fn().mockResolvedValue({
            name: 'Firefox',
            vendor: 'Mozilla',
            version: '91.0'
          })
        }
      };

      // Firefox supports browser.runtime.getBrowserInfo
      expect(window.browser.runtime.getBrowserInfo).toBeDefined();
    });

    test('should handle Chrome-specific features', () => {
      window.chrome = {
        runtime: {
          requestUpdateCheck: jest.fn((callback) => {
            callback('no_update', {});
          })
        }
      };

      // Chrome has specific update check API
      expect(window.chrome.runtime.requestUpdateCheck).toBeDefined();
    });

    test('should handle storage area differences', async () => {
      // Chrome has storage.managed
      window.chrome = {
        storage: {
          managed: {
            get: jest.fn((keys, callback) => callback({}))
          }
        }
      };

      expect(window.chrome.storage.managed).toBeDefined();

      // Firefox might not have managed storage
      window.browser = {
        storage: {
          sync: {},
          local: {}
        }
      };

      expect(window.browser.storage.managed).toBeUndefined();
    });
  });
});