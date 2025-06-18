// Jest setup file
import 'jest-chrome';
import fetchMock from 'jest-fetch-mock';

// Enable fetch mocks
fetchMock.enableMocks();

// Setup Chrome API mocks
global.chrome = {
  storage: {
    sync: {
      get: jest.fn((keys, callback) => {
        const defaults = {
          serverUrl: 'http://localhost:8000',
          apiToken: 'test-token',
          defaultModel: 'gpt-4',
          defaultTemperature: 0.7,
          maxTokens: 1000
        };
        if (callback) {
          callback(defaults);
        }
        return Promise.resolve(defaults);
      }),
      set: jest.fn((data, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      clear: jest.fn((callback) => {
        if (callback) callback();
        return Promise.resolve();
      })
    },
    local: {
      get: jest.fn((keys, callback) => {
        if (callback) callback({});
        return Promise.resolve({});
      }),
      set: jest.fn((data, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      clear: jest.fn((callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      remove: jest.fn((keys, callback) => {
        if (callback) callback();
        return Promise.resolve();
      })
    }
  },
  runtime: {
    onMessage: {
      addListener: jest.fn()
    },
    onInstalled: {
      addListener: jest.fn()
    },
    sendMessage: jest.fn((message, callback) => {
      if (callback) callback({ success: true });
      return Promise.resolve({ success: true });
    }),
    openOptionsPage: jest.fn()
  },
  tabs: {
    query: jest.fn(() => Promise.resolve([{ id: 1, url: 'https://example.com' }])),
    sendMessage: jest.fn((tabId, message, callback) => {
      if (callback) callback({ success: true });
      return Promise.resolve({ success: true });
    }),
    create: jest.fn()
  },
  contextMenus: {
    create: jest.fn(),
    onClicked: {
      addListener: jest.fn()
    }
  },
  action: {
    openPopup: jest.fn(),
    setBadgeText: jest.fn(),
    setBadgeBackgroundColor: jest.fn()
  },
  commands: {
    onCommand: {
      addListener: jest.fn()
    }
  },
  notifications: {
    create: jest.fn()
  },
  downloads: {
    download: jest.fn()
  }
};

// Reset mocks before each test
beforeEach(() => {
  fetchMock.resetMocks();
  jest.clearAllMocks();
});

// Suppress console errors during tests
global.console = {
  ...console,
  error: jest.fn(),
  warn: jest.fn()
};