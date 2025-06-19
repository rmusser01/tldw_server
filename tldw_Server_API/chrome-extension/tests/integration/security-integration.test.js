// Integration tests for API Security and CORS
describe('API Security Integration Tests', () => {
  let apiClient;
  let mockFetch;
  let mockConfigManager;

  beforeEach(async () => {
    // Mock fetch globally
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    global.AbortController = jest.fn().mockImplementation(() => ({
      signal: {},
      abort: jest.fn()
    }));

    // Mock configuration manager
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
          'enableCaching': true,
          'allowedOrigins': ['http://localhost:8000']
        };
        return config[key] ?? defaultValue;
      }),
      isDevelopmentMode: jest.fn().mockReturnValue(true),
      addListener: jest.fn(),
      initialize: jest.fn().mockResolvedValue()
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
            apiToken: 'test-token-123'
          })
        }
      }
    };

    global.window = {
      configManager: mockConfigManager,
      location: { origin: 'chrome-extension://test-extension' }
    };

    global.navigator = {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    };

    // Clear timeout mock
    global.setTimeout = jest.fn().mockImplementation((fn, delay) => {
      return { id: Math.random(), fn, delay };
    });
    global.clearTimeout = jest.fn();

    // Initialize API client
    const TLDWApiClient = require('../../js/utils/api.js');
    apiClient = new TLDWApiClient();
    await apiClient.init();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Request Security Headers', () => {
    test('should include comprehensive security headers in all requests', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };
      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.request('/test');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/test',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Cache-Control': 'no-cache',
            'X-Requested-With': 'XMLHttpRequest',
            'Authorization': 'Bearer test-token-123',
            'User-Agent': expect.stringMatching(/TLDW-Extension\/1\.0\.0/),
            'X-Request-ID': expect.stringMatching(/^req_\d+_[a-z0-9]+$/),
            'Access-Control-Request-Method': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Request-Headers': 'Content-Type, Authorization, X-Requested-With'
          })
        })
      );
    });

    test('should use custom API key header when configured', async () => {
      mockConfigManager.get.mockImplementation((key, defaultValue) => {
        if (key === 'apiKeyHeader') return 'X-API-Key';
        if (key === 'enableCORS') return true;
        return defaultValue;
      });

      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };
      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.request('/test');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-API-Key': 'Bearer test-token-123'
          })
        })
      );

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.not.objectContaining({
            'Authorization': expect.any(String)
          })
        })
      );
    });
  });

  describe('CORS Preflight Integration', () => {
    test('should send preflight request for complex methods', async () => {
      const preflightResponse = { ok: true };
      const actualResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ success: true })
      };

      mockFetch
        .mockResolvedValueOnce(preflightResponse) // Preflight
        .mockResolvedValueOnce(actualResponse);   // Actual request

      await apiClient.request('/test', { method: 'PUT' });

      expect(mockFetch).toHaveBeenCalledTimes(2);
      
      // First call should be OPTIONS (preflight)
      expect(mockFetch).toHaveBeenNthCalledWith(1, 
        'http://localhost:8000/api/v1/test',
        expect.objectContaining({
          method: 'OPTIONS',
          headers: expect.objectContaining({
            'Access-Control-Request-Method': 'PUT',
            'Origin': 'chrome-extension://test-extension'
          })
        })
      );

      // Second call should be the actual PUT request
      expect(mockFetch).toHaveBeenNthCalledWith(2,
        'http://localhost:8000/api/v1/test',
        expect.objectContaining({
          method: 'PUT'
        })
      );
    });

    test('should handle preflight failures gracefully', async () => {
      const preflightError = new Error('CORS preflight failed');
      const actualResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ success: true })
      };

      mockFetch
        .mockRejectedValueOnce(preflightError)  // Preflight fails
        .mockResolvedValueOnce(actualResponse); // Actual request succeeds

      const result = await apiClient.request('/test', { method: 'PUT' });

      expect(result).toEqual({ success: true });
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    test('should not send preflight for simple requests', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };
      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.request('/test', { method: 'GET' });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'GET'
        })
      );
    });
  });

  describe('Error Handling and Retry Logic', () => {
    test('should categorize and retry network errors', async () => {
      const networkError = new Error('Failed to fetch');
      const successResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'success' })
      };

      mockFetch
        .mockRejectedValueOnce(networkError)    // First attempt fails
        .mockRejectedValueOnce(networkError)    // Second attempt fails  
        .mockResolvedValueOnce(successResponse); // Third attempt succeeds

      const result = await apiClient.request('/test');

      expect(result).toEqual({ data: 'success' });
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    test('should not retry authentication errors', async () => {
      const authErrorResponse = {
        ok: false,
        status: 401,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ detail: 'Unauthorized' })
      };

      mockFetch.mockResolvedValueOnce(authErrorResponse);

      await expect(apiClient.request('/test')).rejects.toThrow();
      expect(mockFetch).toHaveBeenCalledTimes(1); // No retries
    });

    test('should retry rate limit errors with backoff', async () => {
      const rateLimitResponse = {
        ok: false,
        status: 429,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ detail: 'Rate limit exceeded' })
      };
      const successResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'success' })
      };

      mockFetch
        .mockResolvedValueOnce(rateLimitResponse) // Rate limited
        .mockResolvedValueOnce(successResponse);   // Success after retry

      const result = await apiClient.request('/test');

      expect(result).toEqual({ data: 'success' });
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    test('should create enhanced errors with security context', async () => {
      const errorResponse = {
        ok: false,
        status: 403,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ detail: 'Access denied' })
      };

      mockFetch.mockResolvedValueOnce(errorResponse);

      try {
        await apiClient.request('/test');
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.status).toBe(403);
        expect(error.category).toBe('authorization');
        expect(error.userMessage).toContain('Access denied');
        expect(error.endpoint).toBe('/test');
        expect(error.requestId).toMatch(/^req_\d+_[a-z0-9]+$/);
        expect(error.timestamp).toBeDefined();
      }
    });
  });

  describe('Timeout and AbortController Integration', () => {
    test('should abort requests that exceed timeout', async () => {
      const abortController = {
        signal: { aborted: false },
        abort: jest.fn()
      };
      global.AbortController = jest.fn().mockReturnValue(abortController);

      mockConfigManager.getApiTimeout.mockReturnValue(5000);

      // Mock a hanging request
      mockFetch.mockImplementation(() => new Promise(() => {}));

      // Mock setTimeout to immediately call the timeout callback
      global.setTimeout = jest.fn().mockImplementation((callback, delay) => {
        setTimeout(() => callback(), 0);
        return 'timeout-id';
      });

      await expect(apiClient.request('/test')).rejects.toThrow();
      expect(abortController.abort).toHaveBeenCalled();
    });

    test('should clear timeout on successful response', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };

      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.request('/test');

      expect(global.clearTimeout).toHaveBeenCalled();
    });
  });

  describe('Response Validation and Parsing', () => {
    test('should validate JSON responses in development mode', async () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      const mockResponse = {
        ok: true,
        url: 'http://localhost:8000/api/v1/test',
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ error: 'something', detail: undefined })
      };

      mockFetch.mockResolvedValueOnce(mockResponse);
      mockConfigManager.isDevelopmentMode.mockReturnValue(true);

      await apiClient.request('/test');

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Non-standard error format'),
        expect.any(Object)
      );

      consoleSpy.mockRestore();
    });

    test('should handle malformed JSON responses', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockRejectedValue(new Error('Unexpected token'))
      };

      mockFetch.mockResolvedValueOnce(mockResponse);

      await expect(apiClient.request('/test')).rejects.toThrow('Invalid response format from server');
    });

    test('should parse different content types correctly', async () => {
      // Test plain text response
      const textResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('text/plain') },
        text: jest.fn().mockResolvedValue('plain text response')
      };

      mockFetch.mockResolvedValueOnce(textResponse);

      const result = await apiClient.request('/test');
      expect(result).toBe('plain text response');
    });
  });

  describe('Configuration Integration', () => {
    test('should respect CORS disable setting', async () => {
      mockConfigManager.get.mockImplementation((key, defaultValue) => {
        if (key === 'enableCORS') return false;
        return defaultValue;
      });

      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };
      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.request('/test');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.not.objectContaining({
            'Access-Control-Request-Method': expect.any(String),
            'Access-Control-Request-Headers': expect.any(String)
          })
        })
      );
    });

    test('should use different timeout values per request type', async () => {
      // Short timeout for connection check
      mockConfigManager.getApiTimeout.mockReturnValue(45000);

      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };
      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.checkConnection();

      // Should use min of configured timeout and 5000ms for connection check
      expect(global.setTimeout).toHaveBeenCalledWith(expect.any(Function), 5000);
    });
  });

  describe('Request Deduplication and Caching', () => {
    test('should deduplicate identical GET requests', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'cached' })
      };

      mockFetch.mockResolvedValue(mockResponse);

      // Make two identical requests simultaneously
      const [result1, result2] = await Promise.all([
        apiClient.request('/prompts/'),
        apiClient.request('/prompts/')
      ]);

      expect(result1).toEqual(result2);
      expect(mockFetch).toHaveBeenCalledTimes(1); // Only one actual request
    });

    test('should cache GET responses when caching enabled', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'cached' })
      };

      mockFetch.mockResolvedValue(mockResponse);

      // First request
      const result1 = await apiClient.request('/prompts/');
      
      // Second request should use cache
      const result2 = await apiClient.request('/prompts/');

      expect(result1).toEqual(result2);
      expect(mockFetch).toHaveBeenCalledTimes(1); // Only one actual request
    });

    test('should invalidate cache on POST/PUT/DELETE requests', async () => {
      const getResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'original' })
      };

      const postResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ success: true })
      };

      const updatedGetResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'updated' })
      };

      mockFetch
        .mockResolvedValueOnce(getResponse)        // Initial GET
        .mockResolvedValueOnce(postResponse)       // POST request
        .mockResolvedValueOnce(updatedGetResponse); // GET after POST

      // Cache initial GET
      await apiClient.request('/prompts/');
      
      // POST should invalidate cache
      await apiClient.request('/prompts/', { method: 'POST' });
      
      // Subsequent GET should fetch fresh data
      const result = await apiClient.request('/prompts/');

      expect(result).toEqual({ data: 'updated' });
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
  });

  describe('Connection Status Integration', () => {
    test('should update connection status on successful requests', async () => {
      const mockResponse = {
        ok: true,
        headers: { get: jest.fn().mockReturnValue('application/json') },
        json: jest.fn().mockResolvedValue({ data: 'test' })
      };

      mockFetch.mockResolvedValueOnce(mockResponse);

      await apiClient.request('/test');

      expect(apiClient.connectionStatus.isConnected).toBe(true);
      expect(apiClient.connectionStatus.consecutiveFailures).toBe(0);
    });

    test('should update connection status on failed requests', async () => {
      const networkError = new Error('Failed to fetch');
      mockFetch.mockRejectedValue(networkError);

      // Disable retries for this test
      mockConfigManager.get.mockImplementation((key, defaultValue) => {
        if (key === 'maxRetries') return 0;
        return defaultValue;
      });

      await expect(apiClient.request('/test')).rejects.toThrow();

      expect(apiClient.connectionStatus.isConnected).toBe(false);
      expect(apiClient.connectionStatus.consecutiveFailures).toBeGreaterThan(0);
    });
  });
});

// Cross-browser compatibility tests
describe('Cross-Browser Security Compatibility', () => {
  test('should work with Firefox browser API', async () => {
    // Mock Firefox environment
    delete global.chrome;
    global.browser = {
      runtime: {
        getManifest: jest.fn().mockReturnValue({ version: '1.0.0' })
      },
      storage: {
        sync: {
          get: jest.fn().mockResolvedValue({
            serverUrl: 'http://localhost:8000',
            apiToken: 'firefox-token'
          })
        }
      }
    };

    global.navigator = {
      userAgent: 'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
    };

    const TLDWApiClient = require('../../js/utils/api.js');
    const firefoxApiClient = new TLDWApiClient();
    await firefoxApiClient.init();

    const userAgent = firefoxApiClient.getUserAgent();
    expect(userAgent).toContain('Firefox/89.0');
    expect(firefoxApiClient.apiToken).toBe('firefox-token');
  });

  test('should handle Edge browser detection', () => {
    global.chrome = {
      runtime: { getManifest: jest.fn().mockReturnValue({ version: '1.0.0' }) },
      storage: { sync: { get: jest.fn().mockResolvedValue({}) } }
    };

    global.navigator = {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
    };

    const TLDWApiClient = require('../../js/utils/api.js');
    const edgeApiClient = new TLDWApiClient();

    const browserInfo = edgeApiClient.getBrowserInfo();
    expect(browserInfo).toContain('Chrome/91.0.4472.124'); // Edge uses Chrome engine
  });
});