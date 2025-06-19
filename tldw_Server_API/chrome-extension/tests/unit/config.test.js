// Unit tests for Configuration Management System
describe('ConfigManager', () => {
  let configManager;
  let mockStorage;

  beforeEach(() => {
    // Mock chrome storage
    mockStorage = {
      sync: {
        get: jest.fn(),
        set: jest.fn(),
      },
      local: {
        get: jest.fn(),
        set: jest.fn(),
        clear: jest.fn(),
        remove: jest.fn(),
      }
    };

    global.chrome = {
      storage: mockStorage,
      runtime: {
        getManifest: jest.fn().mockReturnValue({ version: '1.0.0' })
      }
    };

    // Mock window.location for environment detection
    delete window.location;
    window.location = { hostname: 'localhost' };

    const { ConfigManager } = require('../../js/utils/config.js');
    configManager = new ConfigManager();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('initialization', () => {
    test('should initialize with default configuration', async () => {
      mockStorage.sync.get.mockResolvedValue({});
      
      await configManager.initialize();
      
      expect(configManager.get('serverUrl')).toBe('http://localhost:8000');
      expect(configManager.get('environment')).toBe('development');
      expect(configManager.get('debug')).toBe(true);
    });

    test('should merge user settings with defaults', async () => {
      const userSettings = {
        serverUrl: 'https://custom.server.com',
        apiTimeout: 60000
      };
      
      mockStorage.sync.get.mockResolvedValue(userSettings);
      
      await configManager.initialize();
      
      expect(configManager.get('serverUrl')).toBe('https://custom.server.com');
      expect(configManager.get('apiTimeout')).toBe(60000);
      expect(configManager.get('enableCaching')).toBe(true); // default value
    });

    test('should detect production environment correctly', async () => {
      window.location.hostname = 'api.tldw.example.com';
      mockStorage.sync.get.mockResolvedValue({});
      
      await configManager.initialize();
      
      expect(configManager.get('environment')).toBe('production');
      expect(configManager.get('debug')).toBe(false);
    });
  });

  describe('configuration validation', () => {
    test('should validate server URL and use default for invalid URLs', async () => {
      mockStorage.sync.get.mockResolvedValue({
        serverUrl: 'invalid-url'
      });
      
      await configManager.initialize();
      
      expect(configManager.get('serverUrl')).toBe('http://localhost:8000');
    });

    test('should validate numeric values', async () => {
      mockStorage.sync.get.mockResolvedValue({
        apiTimeout: 'invalid',
        maxRetries: -1
      });
      
      await configManager.initialize();
      
      expect(configManager.get('apiTimeout')).toBe(30000);
      expect(configManager.get('maxRetries')).toBe(3);
    });
  });

  describe('configuration management', () => {
    beforeEach(async () => {
      mockStorage.sync.get.mockResolvedValue({});
      await configManager.initialize();
    });

    test('should set and get configuration values', () => {
      configManager.set('customKey', 'customValue');
      
      expect(configManager.get('customKey')).toBe('customValue');
      expect(mockStorage.sync.set).toHaveBeenCalledWith({ customKey: 'customValue' });
    });

    test('should return default value for unknown keys', () => {
      expect(configManager.get('unknownKey', 'defaultValue')).toBe('defaultValue');
    });

    test('should construct API URLs correctly', () => {
      const apiUrl = configManager.getApiUrl('/test');
      expect(apiUrl).toBe('http://localhost:8000/api/v1/test');
    });

    test('should check feature flags', () => {
      expect(configManager.isFeatureEnabled('enableSmartContext')).toBe(true);
      expect(configManager.isFeatureEnabled('nonExistentFeature')).toBe(false);
    });
  });

  describe('presets', () => {
    beforeEach(async () => {
      mockStorage.sync.get.mockResolvedValue({});
      await configManager.initialize();
    });

    test('should apply performance preset', () => {
      configManager.applyPreset('performance');
      
      expect(configManager.get('enableCaching')).toBe(true);
      expect(configManager.get('cacheTimeout')).toBe(600000);
      expect(configManager.get('batchConcurrency')).toBe(5);
    });

    test('should apply minimal preset', () => {
      configManager.applyPreset('minimal');
      
      expect(configManager.get('enableSmartContext')).toBe(false);
      expect(configManager.get('enableBatchOperations')).toBe(false);
    });
  });

  describe('export/import', () => {
    beforeEach(async () => {
      mockStorage.sync.get.mockResolvedValue({});
      await configManager.initialize();
    });

    test('should export configuration', () => {
      configManager.set('testKey', 'testValue');
      
      const exportedConfig = configManager.exportConfig();
      const parsed = JSON.parse(exportedConfig);
      
      expect(parsed.version).toBe('1.0');
      expect(parsed.config.testKey).toBe('testValue');
      expect(parsed.timestamp).toBeDefined();
    });

    test('should import configuration', async () => {
      const configToImport = {
        version: '1.0',
        timestamp: new Date().toISOString(),
        config: {
          serverUrl: 'https://imported.server.com',
          apiTimeout: 45000
        }
      };
      
      await configManager.importConfig(JSON.stringify(configToImport));
      
      expect(configManager.get('serverUrl')).toBe('https://imported.server.com');
      expect(configManager.get('apiTimeout')).toBe(45000);
    });

    test('should fail import with invalid JSON', async () => {
      await expect(configManager.importConfig('invalid json'))
        .rejects.toThrow();
    });
  });

  describe('health check', () => {
    beforeEach(async () => {
      mockStorage.sync.get.mockResolvedValue({});
      await configManager.initialize();
    });

    test('should pass health check with valid configuration', () => {
      const health = configManager.healthCheck();
      
      expect(health.healthy).toBe(true);
      expect(health.issues).toHaveLength(0);
    });

    test('should detect configuration issues', () => {
      configManager.set('serverUrl', 'invalid-url');
      configManager.set('apiTimeout', 500);
      
      const health = configManager.healthCheck();
      
      expect(health.healthy).toBe(false);
      expect(health.issues).toContain('Invalid server URL');
      expect(health.issues).toContain('API timeout too low (minimum 1000ms recommended)');
    });
  });

  describe('event system', () => {
    beforeEach(async () => {
      mockStorage.sync.get.mockResolvedValue({});
      await configManager.initialize();
    });

    test('should notify listeners on configuration changes', (done) => {
      const listener = jest.fn((event, data) => {
        if (event === 'updated') {
          expect(data.testKey).toBe('testValue');
          done();
        }
      });
      
      configManager.addListener(listener);
      configManager.set('testKey', 'testValue');
    });

    test('should remove listeners correctly', () => {
      const listener = jest.fn();
      const removeListener = configManager.addListener(listener);
      
      removeListener();
      configManager.set('testKey', 'testValue');
      
      expect(listener).not.toHaveBeenCalled();
    });
  });
});

// Property-based tests for configuration validation
describe('ConfigManager Property Tests', () => {
  const fc = require('fast-check');
  let configManager;

  beforeEach(() => {
    global.chrome = {
      storage: {
        sync: { get: jest.fn(), set: jest.fn() },
        local: { get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() }
      },
      runtime: { getManifest: jest.fn().mockReturnValue({ version: '1.0.0' }) }
    };

    window.location = { hostname: 'localhost' };
    
    const { ConfigManager } = require('../../js/utils/config.js');
    configManager = new ConfigManager();
  });

  test('URL validation should handle arbitrary strings', () => {
    fc.assert(fc.property(fc.string(), (url) => {
      const isValid = configManager.isValidUrl(url);
      
      if (isValid) {
        // If validation passes, URL constructor should not throw
        expect(() => new URL(url)).not.toThrow();
      }
    }));
  });

  test('Configuration values should maintain type consistency', async () => {
    chrome.storage.sync.get.mockResolvedValue({});
    await configManager.initialize();

    fc.assert(fc.property(
      fc.oneof(fc.string(), fc.integer(), fc.boolean(), fc.object()),
      (value) => {
        const key = 'testKey';
        configManager.set(key, value);
        
        const retrieved = configManager.get(key);
        expect(retrieved).toEqual(value);
      }
    ));
  });

  test('Version comparison should be transitive', () => {
    fc.assert(fc.property(
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      ([a1, a2, a3], [b1, b2, b3], [c1, c2, c3]) => {
        const versionA = `${a1}.${a2}.${a3}`;
        const versionB = `${b1}.${b2}.${b3}`;
        const versionC = `${c1}.${c2}.${c3}`;
        
        const compAB = configManager.compareVersions(versionA, versionB);
        const compBC = configManager.compareVersions(versionB, versionC);
        const compAC = configManager.compareVersions(versionA, versionC);
        
        // Transitivity: if A >= B and B >= C, then A >= C
        if (compAB >= 0 && compBC >= 0) {
          expect(compAC).toBeGreaterThanOrEqual(0);
        }
      }
    ));
  });
});