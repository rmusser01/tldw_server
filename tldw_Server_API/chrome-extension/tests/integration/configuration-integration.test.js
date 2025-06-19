// Integration tests for Configuration Management System
describe('Configuration Integration Tests', () => {
  let configManager;
  let apiClient;
  let mockStorage;

  beforeEach(async () => {
    // Setup mock Chrome storage
    mockStorage = {
      sync: {
        data: {},
        get: jest.fn().mockImplementation((keys) => {
          if (typeof keys === 'string') {
            return Promise.resolve({ [keys]: mockStorage.sync.data[keys] });
          } else if (Array.isArray(keys)) {
            const result = {};
            keys.forEach(key => {
              if (key in mockStorage.sync.data) {
                result[key] = mockStorage.sync.data[key];
              }
            });
            return Promise.resolve(result);
          } else {
            return Promise.resolve({ ...mockStorage.sync.data });
          }
        }),
        set: jest.fn().mockImplementation((items) => {
          Object.assign(mockStorage.sync.data, items);
          return Promise.resolve();
        }),
        clear: jest.fn().mockImplementation(() => {
          mockStorage.sync.data = {};
          return Promise.resolve();
        }),
        remove: jest.fn().mockImplementation((keys) => {
          const keysArray = Array.isArray(keys) ? keys : [keys];
          keysArray.forEach(key => delete mockStorage.sync.data[key]);
          return Promise.resolve();
        })
      },
      local: {
        data: {},
        get: jest.fn().mockImplementation(() => Promise.resolve({ ...mockStorage.local.data })),
        set: jest.fn().mockImplementation((items) => {
          Object.assign(mockStorage.local.data, items);
          return Promise.resolve();
        }),
        clear: jest.fn().mockImplementation(() => {
          mockStorage.local.data = {};
          return Promise.resolve();
        }),
        remove: jest.fn().mockImplementation((keys) => {
          const keysArray = Array.isArray(keys) ? keys : [keys];
          keysArray.forEach(key => delete mockStorage.local.data[key]);
          return Promise.resolve();
        })
      }
    };

    global.chrome = {
      storage: mockStorage,
      runtime: {
        getManifest: jest.fn().mockReturnValue({ version: '1.0.0' })
      }
    };

    global.window = { location: { hostname: 'localhost' } };
    global.localStorage = {
      getItem: jest.fn(),
      setItem: jest.fn(),
      removeItem: jest.fn()
    };

    // Initialize configuration manager
    const { ConfigManager } = require('../../js/utils/config.js');
    configManager = new ConfigManager();
    await configManager.initialize();

    // Initialize API client with config manager
    global.window.configManager = configManager;
    const TLDWApiClient = require('../../js/utils/api.js');
    apiClient = new TLDWApiClient();
    await apiClient.init();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Config Manager and API Client Integration', () => {
    test('should use config manager for API base URL', () => {
      configManager.set('serverUrl', 'https://custom.api.com');
      
      expect(configManager.getServerUrl()).toBe('https://custom.api.com');
      expect(configManager.getApiUrl('/test')).toBe('https://custom.api.com/api/v1/test');
    });

    test('should propagate configuration changes to API client', async () => {
      const originalUrl = apiClient.baseUrl;
      
      // Change server URL through config manager
      await configManager.saveUserSettings({ serverUrl: 'https://new.server.com' });
      
      // Reinitialize API client to pick up changes
      await apiClient.init();
      
      expect(apiClient.baseUrl).toBe('https://new.server.com');
      expect(apiClient.baseUrl).not.toBe(originalUrl);
    });

    test('should use config manager timeout values in API requests', () => {
      configManager.set('apiTimeout', 45000);
      
      expect(apiClient.getApiTimeout()).toBe(45000);
    });

    test('should use config manager retry settings', () => {
      configManager.set('maxRetries', 5);
      configManager.set('retryDelay', 2000);
      configManager.set('retryBackoffMultiplier', 3);
      
      const retryConfig = apiClient.getRetryConfig();
      
      expect(retryConfig.maxRetries).toBe(5);
      expect(retryConfig.baseDelay).toBe(2000);
      expect(retryConfig.maxDelay).toBe(18000); // 2000 * 3^3
    });

    test('should use config manager cache settings', () => {
      configManager.set('enableCaching', false);
      configManager.set('cacheTimeout', 120000);
      configManager.set('maxCacheSize', 50);
      
      const cacheConfig = apiClient.getCacheConfig();
      
      expect(cacheConfig.enabled).toBe(false);
      expect(cacheConfig.defaultTTL).toBe(120000);
      expect(cacheConfig.maxCacheSize).toBe(50);
    });
  });

  describe('Environment-Specific Configuration', () => {
    test('should apply development environment settings', async () => {
      global.window.location.hostname = 'localhost';
      
      const devConfigManager = new (require('../../js/utils/config.js')).ConfigManager();
      await devConfigManager.initialize();
      
      expect(devConfigManager.get('environment')).toBe('development');
      expect(devConfigManager.get('debug')).toBe(true);
      expect(devConfigManager.get('logLevel')).toBe('debug');
    });

    test('should apply production environment settings', async () => {
      global.window.location.hostname = 'api.tldw.example.com';
      
      const prodConfigManager = new (require('../../js/utils/config.js')).ConfigManager();
      await prodConfigManager.initialize();
      
      expect(prodConfigManager.get('environment')).toBe('production');
      expect(prodConfigManager.get('debug')).toBe(false);
      expect(prodConfigManager.get('logLevel')).toBe('warn');
      expect(prodConfigManager.get('cacheTimeout')).toBe(600000);
    });
  });

  describe('Persistent Storage Integration', () => {
    test('should persist configuration changes across sessions', async () => {
      // Set some configuration
      await configManager.saveUserSettings({
        serverUrl: 'https://persistent.server.com',
        enableSmartContext: false,
        customSetting: 'test'
      });
      
      // Verify it's stored in Chrome storage
      expect(mockStorage.sync.data).toMatchObject({
        serverUrl: 'https://persistent.server.com',
        enableSmartContext: false,
        customSetting: 'test'
      });
      
      // Create new config manager instance (simulating restart)
      const newConfigManager = new (require('../../js/utils/config.js')).ConfigManager();
      await newConfigManager.initialize();
      
      // Verify settings were loaded
      expect(newConfigManager.get('serverUrl')).toBe('https://persistent.server.com');
      expect(newConfigManager.get('enableSmartContext')).toBe(false);
      expect(newConfigManager.get('customSetting')).toBe('test');
    });

    test('should handle storage errors gracefully', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      // Mock storage error
      mockStorage.sync.set.mockRejectedValueOnce(new Error('Storage quota exceeded'));
      
      await expect(configManager.saveUserSettings({ test: 'value' }))
        .rejects.toThrow('Storage quota exceeded');
      
      consoleSpy.mockRestore();
    });

    test('should fall back to localStorage when Chrome storage unavailable', async () => {
      // Remove Chrome storage
      delete global.chrome.storage;
      global.localStorage.getItem.mockReturnValue(JSON.stringify({
        serverUrl: 'https://fallback.server.com'
      }));
      
      const fallbackConfigManager = new (require('../../js/utils/config.js')).ConfigManager();
      await fallbackConfigManager.initialize();
      
      expect(fallbackConfigManager.get('serverUrl')).toBe('https://fallback.server.com');
      
      // Test saving
      await fallbackConfigManager.saveUserSettings({ newSetting: 'value' });
      
      expect(global.localStorage.setItem).toHaveBeenCalledWith(
        'tldw-config',
        expect.stringContaining('newSetting')
      );
    });
  });

  describe('Configuration Validation and Health', () => {
    test('should validate and correct invalid configurations', async () => {
      // Set invalid configuration
      await mockStorage.sync.set({
        serverUrl: 'not-a-url',
        apiTimeout: -1000,
        maxRetries: 'invalid'
      });
      
      const validatingConfigManager = new (require('../../js/utils/config.js')).ConfigManager();
      await validatingConfigManager.initialize();
      
      // Should use defaults for invalid values
      expect(validatingConfigManager.get('serverUrl')).toBe('http://localhost:8000');
      expect(validatingConfigManager.get('apiTimeout')).toBe(30000);
      expect(validatingConfigManager.get('maxRetries')).toBe(3);
    });

    test('should report configuration health issues', () => {
      configManager.set('serverUrl', 'invalid-url');
      configManager.set('apiTimeout', 100);
      configManager.set('maxFileSize', 1000 * 1024 * 1024); // 1GB
      
      const health = configManager.healthCheck();
      
      expect(health.healthy).toBe(false);
      expect(health.issues).toContain('Invalid server URL');
      expect(health.issues).toContain('API timeout too low (minimum 1000ms recommended)');
      expect(health.issues).toContain('Max file size too large (500MB maximum recommended)');
    });

    test('should pass health check with valid configuration', () => {
      const health = configManager.healthCheck();
      
      expect(health.healthy).toBe(true);
      expect(health.issues).toHaveLength(0);
      expect(health.config).toBeDefined();
    });
  });

  describe('Configuration Export/Import', () => {
    test('should export and import configuration correctly', async () => {
      // Set up test configuration
      await configManager.saveUserSettings({
        serverUrl: 'https://export.test.com',
        apiToken: 'export-token',
        enableSmartContext: false,
        customSettings: { test: 'value' }
      });
      
      // Export configuration
      const exportedConfig = configManager.exportConfig();
      const exportData = JSON.parse(exportedConfig);
      
      expect(exportData.version).toBe('1.0');
      expect(exportData.config.serverUrl).toBe('https://export.test.com');
      expect(exportData.config.apiToken).toBe('export-token');
      
      // Clear current configuration
      await configManager.resetToDefaults();
      expect(configManager.get('serverUrl')).toBe('http://localhost:8000');
      
      // Import configuration
      await configManager.importConfig(exportedConfig);
      
      expect(configManager.get('serverUrl')).toBe('https://export.test.com');
      expect(configManager.get('apiToken')).toBe('export-token');
      expect(configManager.get('enableSmartContext')).toBe(false);
    });

    test('should handle configuration migration during import', async () => {
      const oldFormatConfig = {
        version: '1.0',
        config: {
          server_url: 'https://old.format.com', // underscore format
          api_key: 'old-key', // different key name
          enable_smart_context: true
        }
      };
      
      // This should be handled by the validation/transformation logic
      await expect(configManager.importConfig(JSON.stringify(oldFormatConfig)))
        .resolves.toBe(true);
    });
  });

  describe('Event System Integration', () => {
    test('should notify listeners when configuration changes', async () => {
      const listener = jest.fn();
      const removeListener = configManager.addListener(listener);
      
      await configManager.saveUserSettings({ testKey: 'testValue' });
      
      expect(listener).toHaveBeenCalledWith('updated', { testKey: 'testValue' });
      
      removeListener();
      
      await configManager.saveUserSettings({ anotherKey: 'anotherValue' });
      expect(listener).toHaveBeenCalledTimes(1); // Should not be called again
    });

    test('should handle listener errors gracefully', async () => {
      const errorListener = jest.fn().mockImplementation(() => {
        throw new Error('Listener error');
      });
      const goodListener = jest.fn();
      
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      configManager.addListener(errorListener);
      configManager.addListener(goodListener);
      
      await configManager.saveUserSettings({ testKey: 'testValue' });
      
      expect(consoleSpy).toHaveBeenCalledWith(
        '[ConfigManager] Listener error:',
        expect.any(Error)
      );
      expect(goodListener).toHaveBeenCalled(); // Good listener should still work
      
      consoleSpy.mockRestore();
    });
  });

  describe('Feature Flag Integration', () => {
    test('should control feature availability through configuration', async () => {
      // Test with features enabled
      await configManager.saveUserSettings({
        enableSmartContext: true,
        enableBatchOperations: true,
        enableAdvancedSearch: true
      });
      
      expect(configManager.isFeatureEnabled('enableSmartContext')).toBe(true);
      expect(configManager.isFeatureEnabled('enableBatchOperations')).toBe(true);
      expect(configManager.isFeatureEnabled('enableAdvancedSearch')).toBe(true);
      
      // Test with features disabled
      await configManager.saveUserSettings({
        enableSmartContext: false,
        enableBatchOperations: false
      });
      
      expect(configManager.isFeatureEnabled('enableSmartContext')).toBe(false);
      expect(configManager.isFeatureEnabled('enableBatchOperations')).toBe(false);
      expect(configManager.isFeatureEnabled('enableAdvancedSearch')).toBe(true); // unchanged
    });

    test('should apply feature presets correctly', async () => {
      // Apply minimal preset
      configManager.applyPreset('minimal');
      
      expect(configManager.isFeatureEnabled('enableSmartContext')).toBe(false);
      expect(configManager.isFeatureEnabled('enableBatchOperations')).toBe(false);
      expect(configManager.isFeatureEnabled('enableAdvancedSearch')).toBe(false);
      expect(configManager.isFeatureEnabled('enableFloatingButton')).toBe(false);
      
      // Verify settings are persisted
      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          enableSmartContext: false,
          enableBatchOperations: false,
          enableAdvancedSearch: false,
          enableFloatingButton: false
        })
      );
    });
  });

  describe('Performance and Caching Integration', () => {
    test('should respect cache settings in API client', async () => {
      // Configure caching
      await configManager.saveUserSettings({
        enableCaching: true,
        cacheTimeout: 60000,
        maxCacheSize: 25
      });
      
      const cacheConfig = apiClient.getCacheConfig();
      
      expect(cacheConfig.enabled).toBe(true);
      expect(cacheConfig.defaultTTL).toBe(60000);
      expect(cacheConfig.maxCacheSize).toBe(25);
    });

    test('should disable caching when configured', async () => {
      await configManager.saveUserSettings({ enableCaching: false });
      
      const cacheConfig = apiClient.getCacheConfig();
      expect(cacheConfig.enabled).toBe(false);
    });
  });
});

// End-to-end integration test
describe('Full Configuration Lifecycle', () => {
  test('should handle complete configuration lifecycle', async () => {
    // Setup
    const mockStorage = {
      sync: { data: {}, get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() },
      local: { data: {}, get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() }
    };
    
    mockStorage.sync.get.mockImplementation(() => Promise.resolve({}));
    mockStorage.sync.set.mockImplementation((items) => {
      Object.assign(mockStorage.sync.data, items);
      return Promise.resolve();
    });
    
    global.chrome = {
      storage: mockStorage,
      runtime: { getManifest: jest.fn().mockReturnValue({ version: '1.0.0' }) }
    };
    global.window = { location: { hostname: 'localhost' } };
    
    // 1. Initialize configuration
    const { ConfigManager } = require('../../js/utils/config.js');
    const configManager = new ConfigManager();
    await configManager.initialize();
    
    // 2. Verify default settings
    expect(configManager.get('serverUrl')).toBe('http://localhost:8000');
    expect(configManager.get('environment')).toBe('development');
    
    // 3. Update configuration
    await configManager.saveUserSettings({
      serverUrl: 'https://production.server.com',
      apiToken: 'prod-token',
      enableSmartContext: false
    });
    
    // 4. Verify changes are persisted
    expect(mockStorage.sync.data.serverUrl).toBe('https://production.server.com');
    expect(mockStorage.sync.data.apiToken).toBe('prod-token');
    
    // 5. Export configuration
    const exportedConfig = configManager.exportConfig();
    const exportData = JSON.parse(exportedConfig);
    expect(exportData.config.serverUrl).toBe('https://production.server.com');
    
    // 6. Reset and import
    await configManager.resetToDefaults();
    expect(configManager.get('serverUrl')).toBe('http://localhost:8000');
    
    await configManager.importConfig(exportedConfig);
    expect(configManager.get('serverUrl')).toBe('https://production.server.com');
    
    // 7. Verify health
    const health = configManager.healthCheck();
    expect(health.healthy).toBe(true);
    
    // 8. Test API client integration
    global.window.configManager = configManager;
    const TLDWApiClient = require('../../js/utils/api.js');
    const apiClient = new TLDWApiClient();
    await apiClient.init();
    
    expect(apiClient.baseUrl).toBe('https://production.server.com');
    expect(apiClient.apiToken).toBe('prod-token');
    
    // 9. Test feature flags
    expect(configManager.isFeatureEnabled('enableSmartContext')).toBe(false);
    
    // 10. Apply preset and verify
    configManager.applyPreset('performance');
    expect(configManager.get('enableCaching')).toBe(true);
    expect(configManager.get('cacheTimeout')).toBe(600000);
  });
});