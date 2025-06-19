// Custom Jest matchers for extension-specific assertions

expect.extend({
  // Check if a value is a valid API response
  toBeValidAPIResponse(received) {
    const pass = received !== null && 
                 typeof received === 'object' &&
                 !received.hasOwnProperty('error');
    
    return {
      pass,
      message: () => pass
        ? `expected ${JSON.stringify(received)} not to be a valid API response`
        : `expected ${JSON.stringify(received)} to be a valid API response`
    };
  },

  // Check if Chrome storage contains a specific value
  async toHaveStoredValue(storage, key, expectedValue) {
    const result = await storage.get(key);
    const actualValue = result[key];
    const pass = JSON.stringify(actualValue) === JSON.stringify(expectedValue);
    
    return {
      pass,
      message: () => pass
        ? `expected storage not to have ${key} = ${JSON.stringify(expectedValue)}`
        : `expected storage to have ${key} = ${JSON.stringify(expectedValue)}, but got ${JSON.stringify(actualValue)}`
    };
  },

  // Check if an element is visible in the DOM
  toBeVisibleInDOM(element) {
    const pass = element &&
                 element.offsetParent !== null &&
                 element.style.display !== 'none' &&
                 element.style.visibility !== 'hidden';
    
    return {
      pass,
      message: () => pass
        ? `expected element not to be visible in DOM`
        : `expected element to be visible in DOM`
    };
  },

  // Check if a toast notification was shown
  toHaveToastNotification(document, type, messagePattern) {
    const toasts = document.querySelectorAll(`.toast.toast-${type}`);
    let found = false;
    
    for (const toast of toasts) {
      const message = toast.querySelector('.toast-message')?.textContent || '';
      if (messagePattern instanceof RegExp ? messagePattern.test(message) : message.includes(messagePattern)) {
        found = true;
        break;
      }
    }
    
    return {
      pass: found,
      message: () => found
        ? `expected not to find ${type} toast with message matching "${messagePattern}"`
        : `expected to find ${type} toast with message matching "${messagePattern}"`
    };
  },

  // Check if a fetch request was made with specific parameters
  toHaveBeenCalledWithRequest(fetchMock, url, options = {}) {
    const calls = fetchMock.mock.calls;
    const found = calls.some(([callUrl, callOptions = {}]) => {
      if (!callUrl.includes(url)) return false;
      
      if (options.method && callOptions.method !== options.method) return false;
      
      if (options.headers) {
        for (const [key, value] of Object.entries(options.headers)) {
          if (callOptions.headers?.[key] !== value) return false;
        }
      }
      
      if (options.body) {
        const expectedBody = typeof options.body === 'string' 
          ? options.body 
          : JSON.stringify(options.body);
        const actualBody = typeof callOptions.body === 'string'
          ? callOptions.body
          : JSON.stringify(callOptions.body);
        if (actualBody !== expectedBody) return false;
      }
      
      return true;
    });
    
    return {
      pass: found,
      message: () => found
        ? `expected fetch not to be called with ${url} and options ${JSON.stringify(options)}`
        : `expected fetch to be called with ${url} and options ${JSON.stringify(options)}`
    };
  },

  // Check if an async function completes within a time limit
  async toCompleteWithin(asyncFn, timeLimit) {
    const start = Date.now();
    try {
      await asyncFn();
      const duration = Date.now() - start;
      const pass = duration <= timeLimit;
      
      return {
        pass,
        message: () => pass
          ? `expected function not to complete within ${timeLimit}ms, but it took ${duration}ms`
          : `expected function to complete within ${timeLimit}ms, but it took ${duration}ms`
      };
    } catch (error) {
      return {
        pass: false,
        message: () => `expected function to complete within ${timeLimit}ms, but it threw: ${error.message}`
      };
    }
  },

  // Check if a value matches a schema
  toMatchSchema(received, schema) {
    const errors = [];
    
    function validate(obj, schemaObj, path = '') {
      for (const [key, expectedType] of Object.entries(schemaObj)) {
        const fullPath = path ? `${path}.${key}` : key;
        
        if (!(key in obj)) {
          errors.push(`Missing required field: ${fullPath}`);
          continue;
        }
        
        const actualType = Array.isArray(obj[key]) ? 'array' : typeof obj[key];
        
        if (typeof expectedType === 'string') {
          if (actualType !== expectedType) {
            errors.push(`${fullPath}: expected ${expectedType}, got ${actualType}`);
          }
        } else if (typeof expectedType === 'object') {
          if (actualType === 'object' && !Array.isArray(obj[key])) {
            validate(obj[key], expectedType, fullPath);
          } else {
            errors.push(`${fullPath}: expected object, got ${actualType}`);
          }
        }
      }
    }
    
    validate(received, schema);
    const pass = errors.length === 0;
    
    return {
      pass,
      message: () => pass
        ? `expected object not to match schema`
        : `expected object to match schema. Errors:\n${errors.join('\n')}`
    };
  }
});

// Export for use in test files
export default expect;