// Test helper functions for common scenarios

// Wait for async operations to complete
export const waitForAsync = (ms = 0) => new Promise(resolve => setTimeout(resolve, ms));

// Setup mock API with predefined responses
export const setupMockAPI = (responses = {}) => {
  const defaultResponses = {
    '/api/v1/health': { status: 'ok' },
    '/api/v1/chat/completions': { choices: [{ message: { content: 'Test response' } }] },
    '/api/v1/prompts': { results: [] },
    '/api/v1/characters': { results: [] },
    '/api/v1/media': { results: [] }
  };
  
  const allResponses = { ...defaultResponses, ...responses };
  
  fetch.mockImplementation((url) => {
    for (const [path, response] of Object.entries(allResponses)) {
      if (url.includes(path)) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(response),
          headers: new Headers({ 'content-type': 'application/json' })
        });
      }
    }
    
    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' })
    });
  });
};

// Setup Chrome storage with initial data
export const setupChromeStorage = (initialData = {}) => {
  const storage = {
    sync: { ...initialData.sync },
    local: { ...initialData.local }
  };
  
  chrome.storage.sync.get.mockImplementation((keys, callback) => {
    const result = {};
    if (Array.isArray(keys)) {
      keys.forEach(key => {
        if (key in storage.sync) result[key] = storage.sync[key];
      });
    } else if (typeof keys === 'object') {
      Object.keys(keys).forEach(key => {
        result[key] = storage.sync[key] !== undefined ? storage.sync[key] : keys[key];
      });
    } else if (keys) {
      if (keys in storage.sync) result[keys] = storage.sync[keys];
    } else {
      Object.assign(result, storage.sync);
    }
    
    if (callback) callback(result);
    return Promise.resolve(result);
  });
  
  chrome.storage.sync.set.mockImplementation((data, callback) => {
    Object.assign(storage.sync, data);
    if (callback) callback();
    return Promise.resolve();
  });
  
  // Similar implementation for local storage
  chrome.storage.local.get.mockImplementation((keys, callback) => {
    const result = {};
    if (Array.isArray(keys)) {
      keys.forEach(key => {
        if (key in storage.local) result[key] = storage.local[key];
      });
    } else if (typeof keys === 'object') {
      Object.keys(keys).forEach(key => {
        result[key] = storage.local[key] !== undefined ? storage.local[key] : keys[key];
      });
    } else if (keys) {
      if (keys in storage.local) result[keys] = storage.local[keys];
    } else {
      Object.assign(result, storage.local);
    }
    
    if (callback) callback(result);
    return Promise.resolve(result);
  });
  
  chrome.storage.local.set.mockImplementation((data, callback) => {
    Object.assign(storage.local, data);
    if (callback) callback();
    return Promise.resolve();
  });
  
  return storage;
};

// Create a mock streaming response
export const createMockStream = (chunks) => {
  let index = 0;
  
  return new ReadableStream({
    start(controller) {
      const pushChunk = () => {
        if (index < chunks.length) {
          const chunk = chunks[index++];
          const encoded = new TextEncoder().encode(
            typeof chunk === 'string' ? chunk : `data: ${JSON.stringify(chunk)}\n\n`
          );
          controller.enqueue(encoded);
          
          if (index < chunks.length) {
            setTimeout(pushChunk, 10);
          } else {
            controller.close();
          }
        }
      };
      
      pushChunk();
    }
  });
};

// Simulate user input
export const simulateUserInput = (element, value) => {
  element.value = value;
  element.dispatchEvent(new Event('input', { bubbles: true }));
  element.dispatchEvent(new Event('change', { bubbles: true }));
};

// Wait for element to appear in DOM
export const waitForElement = (selector, timeout = 5000) => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    const checkElement = () => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
      } else if (Date.now() - startTime > timeout) {
        reject(new Error(`Element ${selector} not found within ${timeout}ms`));
      } else {
        setTimeout(checkElement, 100);
      }
    };
    
    checkElement();
  });
};

// Create DOM structure from HTML string
export const createDOM = (html) => {
  const container = document.createElement('div');
  container.innerHTML = html;
  document.body.appendChild(container);
  return container;
};

// Clean up DOM after tests
export const cleanupDOM = () => {
  document.body.innerHTML = '';
};

// Mock console methods
export const mockConsole = () => {
  const originalConsole = {
    log: console.log,
    error: console.error,
    warn: console.warn,
    info: console.info
  };
  
  const mocks = {
    log: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    info: jest.fn()
  };
  
  Object.assign(console, mocks);
  
  return {
    mocks,
    restore: () => Object.assign(console, originalConsole)
  };
};

// Create a mock for window.api (browser API)
export const createMockBrowserAPI = () => ({
  runtime: {
    openOptionsPage: jest.fn().mockResolvedValue(),
    sendMessage: jest.fn().mockResolvedValue(),
    getURL: jest.fn(path => `chrome-extension://mock-id/${path}`)
  },
  tabs: {
    create: jest.fn().mockResolvedValue({ id: 1 }),
    query: jest.fn().mockResolvedValue([]),
    sendMessage: jest.fn().mockResolvedValue()
  },
  storage: {
    sync: {
      get: jest.fn().mockResolvedValue({}),
      set: jest.fn().mockResolvedValue()
    },
    local: {
      get: jest.fn().mockResolvedValue({}),
      set: jest.fn().mockResolvedValue()
    }
  }
});

// Test error boundary
export const expectToThrowAsync = async (fn, errorPattern) => {
  let error;
  try {
    await fn();
  } catch (e) {
    error = e;
  }
  
  expect(error).toBeDefined();
  if (errorPattern) {
    if (errorPattern instanceof RegExp) {
      expect(error.message).toMatch(errorPattern);
    } else {
      expect(error.message).toContain(errorPattern);
    }
  }
  
  return error;
};

// Performance testing helper
export const measurePerformance = async (fn, iterations = 100) => {
  const times = [];
  
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }
  
  return {
    avg: times.reduce((a, b) => a + b, 0) / times.length,
    min: Math.min(...times),
    max: Math.max(...times),
    median: times.sort((a, b) => a - b)[Math.floor(times.length / 2)]
  };
};