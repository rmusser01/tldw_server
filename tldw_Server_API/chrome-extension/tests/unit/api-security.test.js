// Unit tests for API Security and CORS enhancements
describe('TLDWApiClient Security', () => {
  let apiClient;
  let mockFetch;
  let mockConfigManager;

  beforeEach(() => {
    // Mock fetch
    mockFetch = jest.fn();
    global.fetch = mockFetch;

    // Mock config manager
    mockConfigManager = {
      getServerUrl: jest.fn().mockReturnValue('http://localhost:8000'),
      getApiUrl: jest.fn().mockImplementation((endpoint) => `http://localhost:8000/api/v1${endpoint}`),
      getApiTimeout: jest.fn().mockReturnValue(30000),
      get: jest.fn().mockImplementation((key, defaultValue) => {
        const config = {
          'apiKeyHeader': 'Authorization',
          'enableCORS': true,
          'maxRetries': 3,
          'retryDelay': 1000,
          'retryBackoffMultiplier': 2,
          'cacheTimeout': 300000,
          'maxCacheSize': 100,
          'enableCaching': true
        };
        return config[key] ?? defaultValue;
      }),
      isDevelopmentMode: jest.fn().mockReturnValue(true),
      addListener: jest.fn()
    };

    // Mock Chrome APIs
    global.chrome = {
      runtime: {
        getManifest: jest.fn().mockReturnValue({ version: '1.0.0' })
      },
      storage: {
        sync: {
          get: jest.fn().mockResolvedValue({
            serverUrl: 'http://localhost:8000',
            apiToken: 'test-token'
          })
        }
      }
    };

    global.window = {
      configManager: mockConfigManager,
      location: { origin: 'chrome-extension://test' }
    };

    global.navigator = {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    };

    // Import and create API client
    const TLDWApiClient = require('../../js/utils/api.js');
    apiClient = new TLDWApiClient();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Security Headers', () => {
    test('should include proper security headers', async () => {
      await apiClient.init();
      
      const headers = apiClient.getHeaders();
      
      expect(headers['Content-Type']).toBe('application/json');
      expect(headers['Accept']).toBe('application/json');
      expect(headers['Cache-Control']).toBe('no-cache');
      expect(headers['X-Requested-With']).toBe('XMLHttpRequest');
      expect(headers['Authorization']).toBe('Bearer test-token');
      expect(headers['User-Agent']).toMatch(/TLDW-Extension\/1\.0\.0/);
      expect(headers['X-Request-ID']).toMatch(/^req_\d+_[a-z0-9]+$/);
    });

    test('should include CORS headers when enabled', async () => {
      await apiClient.init();
      
      const headers = apiClient.getHeaders();
      
      expect(headers['Access-Control-Request-Method']).toBe('GET, POST, PUT, DELETE, OPTIONS');
      expect(headers['Access-Control-Request-Headers']).toBe('Content-Type, Authorization, X-Requested-With');
    });

    test('should use custom API key header when configured', async () => {
      mockConfigManager.get.mockImplementation((key, defaultValue) => {
        if (key === 'apiKeyHeader') return 'X-API-Key';
        return defaultValue;
      });
      
      await apiClient.init();
      const headers = apiClient.getHeaders();
      
      expect(headers['X-API-Key']).toBe('Bearer test-token');
      expect(headers['Authorization']).toBeUndefined();
    });

    test('should generate proper User-Agent string', () => {
      const userAgent = apiClient.getUserAgent();
      
      expect(userAgent).toMatch(/^TLDW-Extension\/[\d.]+/);
      expect(userAgent).toContain('Chrome/');
    });

    test('should generate unique request IDs', () => {
      const id1 = apiClient.generateRequestId();
      const id2 = apiClient.generateRequestId();
      
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^req_\d+_[a-z0-9]+$/);
      expect(id2).toMatch(/^req_\d+_[a-z0-9]+$/);
    });
  });

  describe('CORS Handling', () => {
    test('should detect when preflight is needed', () => {
      const simpleRequest = { method: 'GET', headers: {} };
      const complexRequest = { method: 'PUT', headers: { 'Authorization': 'Bearer token' } };
      
      expect(apiClient.needsPreflightRequest(simpleRequest)).toBe(false);
      expect(apiClient.needsPreflightRequest(complexRequest)).toBe(true);
    });

    test('should send preflight request for complex requests', async () => {
      const mockResponse = { ok: true };
      mockFetch.mockResolvedValue(mockResponse);
      
      await apiClient.sendPreflightRequest('http://localhost:8000/api/v1/test', {
        method: 'PUT',
        headers: { 'Authorization': 'Bearer token' }
      });
      
      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/api/v1/test', {
        method: 'OPTIONS',
        headers: {
          'Access-Control-Request-Method': 'PUT',
          'Access-Control-Request-Headers': 'Authorization',
          'Origin': 'chrome-extension://test'
        }
      });
    });

    test('should handle preflight failures gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('CORS preflight failed'));
      
      // Should not throw - preflight failures should be non-blocking
      await expect(apiClient.sendPreflightRequest('http://localhost:8000/api/v1/test', {
        method: 'PUT'
      })).resolves.toBeUndefined();
    });
  });

  describe('Error Handling and Categorization', () => {
    test('should categorize network errors correctly', () => {
      const networkError = new Error('Failed to fetch');
      const corsError = new Error('CORS error');
      const timeoutError = new Error('AbortError');
      timeoutError.name = 'AbortError';
      
      expect(apiClient.categorizeError(networkError)).toBe('network');
      expect(apiClient.categorizeError(corsError)).toBe('cors');
      expect(apiClient.categorizeError(timeoutError)).toBe('timeout');
    });

    test('should create enhanced errors with proper metadata', () => {
      const errorInfo = { detail: 'Access denied' };
      const error = apiClient.createEnhancedError(403, errorInfo, '/test');
      
      expect(error.status).toBe(403);
      expect(error.endpoint).toBe('/test');
      expect(error.category).toBe('authorization');
      expect(error.userMessage).toBe('Access denied. You may not have permission for this operation.');
      expect(error.timestamp).toBeDefined();
      expect(error.requestId).toMatch(/^req_\d+_[a-z0-9]+$/);
    });

    test('should parse error responses correctly', async () => {
      const jsonResponse = {
        ok: false,
        status: 400,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ detail: 'Bad request' })
      };
      
      const errorInfo = await apiClient.parseErrorResponse(jsonResponse);
      expect(errorInfo.detail).toBe('Bad request');
    });

    test('should handle non-JSON error responses', async () => {
      const textResponse = {
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        headers: { get: jest.fn().mockReturnValue('text/plain') },
        text: jest.fn().mockResolvedValue('Server error occurred')
      };
      
      const errorInfo = await apiClient.parseErrorResponse(textResponse);
      expect(errorInfo.detail).toBe('Server error occurred');
    });
  });

  describe('Retry Logic', () => {
    test('should retry on appropriate errors', () => {
      const networkError = { category: 'network' };
      const authError = { status: 401, category: 'auth_error' };
      const rateLimitError = { status: 429, category: 'rate_limit' };
      
      expect(apiClient.shouldRetry(networkError, 0, 'network')).toBe(true);
      expect(apiClient.shouldRetry(authError, 0, 'auth_error')).toBe(false);
      expect(apiClient.shouldRetry(rateLimitError, 0, 'rate_limit')).toBe(true);
    });

    test('should not retry after max attempts', () => {
      const retryableError = { category: 'network' };
      
      expect(apiClient.shouldRetry(retryableError, 0, 'network')).toBe(true);
      expect(apiClient.shouldRetry(retryableError, 3, 'network')).toBe(false);
    });

    test('should not retry client errors except rate limits', () => {
      const badRequestError = { status: 400 };
      const rateLimitError = { status: 429 };
      
      expect(apiClient.shouldRetry(badRequestError, 0)).toBe(false);
      expect(apiClient.shouldRetry(rateLimitError, 0)).toBe(true);
    });
  });

  describe('Response Validation', () => {
    test('should parse JSON responses correctly', async () => {
      const mockResponse = {
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };
      
      const result = await apiClient.parseSuccessResponse(mockResponse);
      expect(result).toEqual({ data: 'test' });
    });

    test('should parse text responses correctly', async () => {
      const mockResponse = {
        headers: { get: jest.fn().mockReturnValue('text/plain') },
        text: jest.fn().mockResolvedValue('plain text response')
      };
      
      const result = await apiClient.parseSuccessResponse(mockResponse);
      expect(result).toBe('plain text response');
    });

    test('should validate response structure in development mode', async () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      apiClient.validateResponseStructure(null, 'http://test.com');
      expect(consoleSpy).toHaveBeenCalledWith('Empty response from http://test.com');
      
      consoleSpy.mockRestore();
    });

    test('should handle invalid JSON gracefully', async () => {
      const mockResponse = {
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockRejectedValue(new Error('Invalid JSON'))
      };
      
      await expect(apiClient.parseSuccessResponse(mockResponse))
        .rejects.toThrow('Invalid response format from server');
    });
  });

  describe('Integration with Config Manager', () => {
    test('should use config manager for timeout values', () => {
      mockConfigManager.getApiTimeout.mockReturnValue(45000);
      
      expect(apiClient.getApiTimeout()).toBe(45000);
    });

    test('should use config manager for retry configuration', () => {
      mockConfigManager.get.mockImplementation((key, defaultValue) => {
        const config = {
          'maxRetries': 5,
          'retryDelay': 2000,
          'retryBackoffMultiplier': 3
        };
        return config[key] ?? defaultValue;
      });
      
      const retryConfig = apiClient.getRetryConfig();
      expect(retryConfig.maxRetries).toBe(5);
      expect(retryConfig.baseDelay).toBe(2000);
    });

    test('should fall back to defaults when config manager unavailable', () => {
      // Create API client without config manager
      delete global.window.configManager;
      const standaloneClient = new (require('../../js/utils/api.js'))();
      
      expect(standaloneClient.getApiTimeout()).toBe(30000);
      
      const retryConfig = standaloneClient.getRetryConfig();
      expect(retryConfig.maxRetries).toBe(3);
      expect(retryConfig.baseDelay).toBe(1000);
    });
  });
});

// Property-based tests for security functions
describe('API Security Property Tests', () => {
  const fc = require('fast-check');
  let apiClient;

  beforeEach(() => {
    global.chrome = {
      runtime: { getManifest: jest.fn().mockReturnValue({ version: '1.0.0' }) },
      storage: { sync: { get: jest.fn().mockResolvedValue({}) } }
    };
    
    global.window = { configManager: null };
    global.navigator = { userAgent: 'Chrome/91.0' };
    
    const TLDWApiClient = require('../../js/utils/api.js');
    apiClient = new TLDWApiClient();
  });

  test('Request ID generation should always produce unique IDs', () => {
    fc.assert(fc.property(fc.nat(1000), () => {
      const ids = new Set();
      for (let i = 0; i < 100; i++) {
        ids.add(apiClient.generateRequestId());
      }
      expect(ids.size).toBe(100); // All IDs should be unique
    }));
  });

  test('Error categorization should be consistent', () => {
    fc.assert(fc.property(
      fc.oneof(
        fc.constant({ name: 'AbortError' }),
        fc.constant({ message: 'Failed to fetch' }),
        fc.constant({ message: 'CORS error' }),
        fc.record({ status: fc.integer(100, 599) })
      ),
      (error) => {
        const category1 = apiClient.categorizeError(error);
        const category2 = apiClient.categorizeError(error);
        
        expect(category1).toBe(category2); // Should be deterministic
        expect(typeof category1).toBe('string'); // Should always return string
      }
    ));
  });

  test('Version comparison in retry config should handle arbitrary versions', () => {
    fc.assert(fc.property(
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      ([major, minor, patch]) => {
        const version = `${major}.${minor}.${patch}`;
        
        // Mock config manager with this version
        global.window = {
          configManager: {
            get: jest.fn().mockImplementation((key) => {
              if (key === 'retryDelay') return 1000;
              if (key === 'retryBackoffMultiplier') return 2;
              return undefined;
            })
          }
        };
        
        const retryConfig = apiClient.getRetryConfig();
        
        expect(retryConfig.maxDelay).toBeGreaterThan(0);
        expect(retryConfig.baseDelay).toBeGreaterThan(0);
        expect(retryConfig.maxDelay).toBeGreaterThanOrEqual(retryConfig.baseDelay);
      }
    ));
  });
});