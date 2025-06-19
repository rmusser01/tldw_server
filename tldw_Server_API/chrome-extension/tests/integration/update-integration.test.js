// Integration tests for Extension Update Management
describe('Extension Update Integration Tests', () => {
  let updateManager;
  let mockChrome;
  let mockStorage;

  beforeEach(() => {
    // Mock Chrome storage with realistic behavior
    mockStorage = {
      sync: {
        data: {},
        get: jest.fn().mockImplementation((keys) => {
          if (!keys) return Promise.resolve({ ...mockStorage.sync.data });
          if (typeof keys === 'string') {
            return Promise.resolve({ [keys]: mockStorage.sync.data[keys] });
          }
          if (Array.isArray(keys)) {
            const result = {};
            keys.forEach(key => {
              if (key in mockStorage.sync.data) {
                result[key] = mockStorage.sync.data[key];
              }
            });
            return Promise.resolve(result);
          }
          return Promise.resolve({});
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
        get: jest.fn().mockImplementation((keys) => {
          if (!keys) return Promise.resolve({ ...mockStorage.local.data });
          if (typeof keys === 'string') {
            return Promise.resolve({ [keys]: mockStorage.local.data[keys] });
          }
          if (Array.isArray(keys)) {
            const result = {};
            keys.forEach(key => {
              if (key in mockStorage.local.data) {
                result[key] = mockStorage.local.data[key];
              }
            });
            return Promise.resolve(result);
          }
          return Promise.resolve({});
        }),
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

    // Mock Chrome APIs
    mockChrome = {
      runtime: {
        getManifest: jest.fn().mockReturnValue({ 
          version: '1.2.0',
          minimum_chrome_version: '88.0'
        }),
        openOptionsPage: jest.fn()
      },
      storage: mockStorage,
      notifications: {
        create: jest.fn().mockImplementation((id, options, callback) => {
          if (callback) callback(id);
        }),
        onClicked: {
          addListener: jest.fn()
        }
      }
    };

    global.chrome = mockChrome;
    global.navigator = {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    };

    // Import and initialize update manager
    const { ExtensionUpdateManager } = require('../../js/background.js');
    updateManager = new ExtensionUpdateManager();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Full Installation Lifecycle', () => {
    test('should handle complete first installation flow', async () => {
      const installDetails = { reason: 'install' };

      await updateManager.handleInstallation(installDetails);

      // Verify default configuration was set
      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: 'http://localhost:8000',
          enableSmartContext: true,
          enableBatchOperations: true,
          enableAdvancedSearch: true,
          enableProgressIndicators: true,
          extensionVersion: '1.2.0',
          firstRun: true
        })
      );

      // Verify local state initialization
      expect(mockStorage.local.clear).toHaveBeenCalled();
      expect(mockStorage.local.set).toHaveBeenCalledWith({
        lastConnectionCheck: null,
        connectionStatus: 'unknown',
        cacheInitialized: false,
        migrationVersion: '1.2.0'
      });

      // Verify welcome notification
      expect(mockChrome.notifications.create).toHaveBeenCalledWith(
        'welcome',
        expect.objectContaining({
          title: 'TLDW Extension Installed',
          message: expect.stringContaining('Welcome')
        }),
        expect.any(Function)
      );

      // Verify options page opened
      expect(mockChrome.runtime.openOptionsPage).toHaveBeenCalled();
    });

    test('should preserve existing configuration during installation', async () => {
      // Set up existing configuration
      mockStorage.sync.data = {
        serverUrl: 'https://custom.server.com',
        apiToken: 'existing-token',
        enableSmartContext: false
      };

      const installDetails = { reason: 'install' };
      await updateManager.handleInstallation(installDetails);

      // Should not override existing values
      expect(mockStorage.sync.data.serverUrl).toBe('https://custom.server.com');
      expect(mockStorage.sync.data.apiToken).toBe('existing-token');
      expect(mockStorage.sync.data.enableSmartContext).toBe(false);

      // Should add missing defaults
      expect(mockStorage.sync.data.enableBatchOperations).toBe(true);
      expect(mockStorage.sync.data.extensionVersion).toBe('1.2.0');
    });
  });

  describe('Update Lifecycle with Migration', () => {
    test('should handle complete update flow with data migration', async () => {
      // Set up existing data before update
      mockStorage.sync.data = {
        serverUrl: 'https://existing.server.com',
        apiToken: 'existing-token',
        oldSetting: 'to-be-migrated'
      };

      mockStorage.local.data = {
        cache_old: 'old-cache-data',
        existingLocalData: 'preserve-this'
      };

      const updateDetails = { reason: 'update', previousVersion: '1.0.0' };

      await updateManager.handleInstallation(updateDetails);

      // Verify backup was created
      const backupKeys = Object.keys(mockStorage.local.data)
        .filter(key => key.startsWith('backup_'));
      expect(backupKeys.length).toBeGreaterThan(0);

      const backupKey = backupKeys[0];
      expect(mockStorage.local.data[backupKey]).toMatchObject({
        version: '1.2.0',
        timestamp: expect.any(String),
        syncData: expect.objectContaining({
          serverUrl: 'https://existing.server.com',
          apiToken: 'existing-token'
        })
      });

      // Verify update notification
      expect(mockChrome.notifications.create).toHaveBeenCalledWith(
        'update',
        expect.objectContaining({
          title: 'TLDW Extension Updated',
          message: expect.stringContaining('v1.0.0 to v1.2.0')
        })
      );

      // Verify version was updated
      expect(mockStorage.sync.data.extensionVersion).toBe('1.2.0');
      expect(mockStorage.sync.data.lastUpdated).toBeDefined();
    });

    test('should execute version-specific migrations in order', async () => {
      // Mock existing data that needs migration
      mockStorage.sync.data = {
        apiKey: 'old-key-format', // Should be migrated to apiToken
        serverUrl: 'https://test.com'
      };

      mockStorage.local.data = { completedMigrations: [] };

      // Simulate update from version that triggers migrations
      const updateDetails = { reason: 'update', previousVersion: '1.1.0' };

      await updateManager.handleInstallation(updateDetails);

      // Check if API key migration occurred (1.1.0->1.2.0 migration)
      expect(mockStorage.sync.data.apiToken).toBe('old-key-format');
      expect(mockStorage.sync.data.apiKey).toBeUndefined();

      // Verify migration was recorded
      expect(mockStorage.local.data.completedMigrations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: 'Migrate API token format',
            version: '1.2.0',
            timestamp: expect.any(String)
          })
        ])
      );
    });

    test('should handle migration failures with rollback', async () => {
      // Set up initial data
      mockStorage.sync.data = {
        serverUrl: 'https://original.server.com',
        importantData: 'preserve-this'
      };

      // Create a backup in local storage
      const backupData = {
        version: '1.1.0',
        timestamp: new Date().toISOString(),
        syncData: { ...mockStorage.sync.data },
        localData: {}
      };
      mockStorage.local.data = { 'backup_1234567890': backupData };

      // Mock a migration that will fail
      const originalMigrationStrategies = updateManager.migrationStrategies;
      updateManager.migrationStrategies = new Map([
        ['1.1.0->1.2.0', {
          name: 'Failing Migration',
          execute: jest.fn().mockRejectedValue(new Error('Migration failed'))
        }]
      ]);

      const updateDetails = { reason: 'update', previousVersion: '1.1.0' };

      // Update should handle the error
      await updateManager.handleInstallation(updateDetails);

      // Verify rollback occurred
      expect(mockStorage.sync.clear).toHaveBeenCalled();
      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: 'https://original.server.com',
          importantData: 'preserve-this'
        })
      );

      // Restore original migration strategies
      updateManager.migrationStrategies = originalMigrationStrategies;
    });
  });

  describe('Backup and Recovery Integration', () => {
    test('should maintain backup rotation correctly', async () => {
      // Create multiple old backups
      const oldBackups = {};
      for (let i = 1; i <= 5; i++) {
        oldBackups[`backup_${1000000 + i}`] = {
          version: '1.0.0',
          timestamp: new Date(Date.now() - i * 86400000).toISOString(), // i days ago
          syncData: { test: `data${i}` }
        };
      }
      mockStorage.local.data = { ...oldBackups, otherData: 'keep-this' };

      await updateManager.backupExtensionData();

      // Should keep only 3 most recent backups plus the new one
      const backupKeys = Object.keys(mockStorage.local.data)
        .filter(key => key.startsWith('backup_'));
      expect(backupKeys.length).toBeLessThanOrEqual(4); // 3 old + 1 new

      // Should not remove non-backup data
      expect(mockStorage.local.data.otherData).toBe('keep-this');

      // Verify oldest backups were removed
      expect(mockStorage.local.remove).toHaveBeenCalledWith(
        expect.arrayContaining(['backup_1000001', 'backup_1000002'])
      );
    });

    test('should handle corrupted backup data gracefully', async () => {
      // Set up corrupted backup data
      mockStorage.local.data = {
        'backup_corrupt': 'not-an-object',
        'backup_valid': {
          version: '1.1.0',
          syncData: { serverUrl: 'https://backup.com' },
          localData: {}
        }
      };

      await updateManager.rollbackOnError();

      // Should use the valid backup
      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: 'https://backup.com'
        })
      );
    });
  });

  describe('Chrome Update Integration', () => {
    test('should handle Chrome browser update correctly', async () => {
      mockStorage.local.data = {
        lastConnectionCheck: Date.now(),
        connectionStatus: 'connected',
        existingCache: 'preserve-this'
      };

      const chromeUpdateDetails = { reason: 'chrome_update' };
      await updateManager.handleInstallation(chromeUpdateDetails);

      // Should reset connection status
      expect(mockStorage.local.set).toHaveBeenCalledWith({
        lastConnectionCheck: null,
        connectionStatus: 'unknown'
      });

      // Should preserve other local data
      expect(mockStorage.local.data.existingCache).toBe('preserve-this');
    });

    test('should verify Chrome version compatibility', async () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      // Mock old Chrome version
      global.navigator.userAgent = 'Chrome/87.0.4280.88';
      mockChrome.runtime.getManifest.mockReturnValue({
        version: '1.2.0',
        minimum_chrome_version: '88.0'
      });

      await updateManager.verifyCompatibility();

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Chrome version 87.0.4280.88 is below minimum required 88.0')
      );

      consoleSpy.mockRestore();
    });
  });

  describe('Cache and Cleanup Integration', () => {
    test('should clean up old cache entries during update', async () => {
      mockStorage.local.data = {
        'cache_old1': 'old-cache-1',
        'cache_old2': 'old-cache-2',
        'temp_data': 'temporary',
        'normal_data': 'keep-this',
        'tempMigrationData': 'remove-this',
        'migrationInProgress': 'remove-this-too'
      };

      await updateManager.cleanupOldData();

      // Should remove cache and temp data
      expect(mockStorage.local.remove).toHaveBeenCalledWith([
        'tempMigrationData', 'migrationInProgress'
      ]);
      expect(mockStorage.local.remove).toHaveBeenCalledWith([
        'cache_old1', 'cache_old2', 'temp_data'
      ]);

      // Should preserve normal data
      expect(mockStorage.local.data.normal_data).toBe('keep-this');
    });

    test('should handle large cache cleanup efficiently', async () => {
      // Create large cache dataset
      const largeCacheData = {};
      for (let i = 0; i < 1000; i++) {
        largeCacheData[`cache_item_${i}`] = `data_${i}`;
        largeCacheData[`temp_item_${i}`] = `temp_${i}`;
      }
      largeCacheData['important_data'] = 'keep-this';

      mockStorage.local.data = largeCacheData;

      await updateManager.cleanupCache();

      // Should remove all cache/temp items efficiently
      expect(mockStorage.local.remove).toHaveBeenCalledWith(
        expect.arrayContaining([
          'cache_item_0', 'temp_item_0', 'cache_item_999', 'temp_item_999'
        ])
      );

      // Should preserve non-cache data
      expect(mockStorage.local.data.important_data).toBe('keep-this');
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should handle storage quota exceeded errors', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      // Mock storage quota error
      mockStorage.sync.set.mockRejectedValue(new Error('QUOTA_BYTES_PER_ITEM quota exceeded'));
      mockStorage.local.set.mockRejectedValue(new Error('QUOTA_BYTES_PER_ITEM quota exceeded'));

      const installDetails = { reason: 'install' };
      await updateManager.handleInstallation(installDetails);

      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to handle first install:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });

    test('should handle concurrent update operations', async () => {
      const updateDetails1 = { reason: 'update', previousVersion: '1.0.0' };
      const updateDetails2 = { reason: 'update', previousVersion: '1.1.0' };

      // Start two update operations simultaneously
      const updatePromise1 = updateManager.handleInstallation(updateDetails1);
      const updatePromise2 = updateManager.handleInstallation(updateDetails2);

      await Promise.all([updatePromise1, updatePromise2]);

      // Should handle both updates gracefully (exact behavior depends on implementation)
      expect(mockStorage.sync.set).toHaveBeenCalled();
    });

    test('should recover from corrupted extension state', async () => {
      // Set up corrupted state
      mockStorage.sync.data = null;
      mockStorage.local.data = { corrupted: 'state' };

      const installDetails = { reason: 'install' };
      await updateManager.handleInstallation(installDetails);

      // Should still complete initialization with defaults
      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: 'http://localhost:8000',
          extensionVersion: '1.2.0'
        })
      );
    });
  });

  describe('Notification Integration', () => {
    test('should handle notification click events for welcome notification', async () => {
      const installDetails = { reason: 'install' };
      await updateManager.handleInstallation(installDetails);

      // Verify notification was created with click handler
      expect(mockChrome.notifications.create).toHaveBeenCalledWith(
        'welcome',
        expect.any(Object),
        expect.any(Function)
      );

      // Verify click listener was registered
      expect(mockChrome.notifications.onClicked.addListener).toHaveBeenCalled();
    });

    test('should show different notifications for different update types', async () => {
      // Test major version update
      const majorUpdateDetails = { reason: 'update', previousVersion: '1.0.0' };
      await updateManager.handleInstallation(majorUpdateDetails);

      expect(mockChrome.notifications.create).toHaveBeenCalledWith(
        'update',
        expect.objectContaining({
          message: expect.stringContaining('v1.0.0 to v1.2.0')
        })
      );
    });
  });

  describe('Version Comparison Edge Cases', () => {
    test('should handle semantic version comparison correctly', () => {
      const testCases = [
        { a: '1.0.0', b: '1.0.0', expected: 0 },
        { a: '1.0.1', b: '1.0.0', expected: 1 },
        { a: '1.0.0', b: '1.0.1', expected: -1 },
        { a: '1.1.0', b: '1.0.9', expected: 1 },
        { a: '2.0.0', b: '1.9.9', expected: 1 },
        { a: '1.0.10', b: '1.0.2', expected: 1 },
        { a: '1.0', b: '1.0.0', expected: 0 },
        { a: '1', b: '1.0.0', expected: 0 }
      ];

      testCases.forEach(({ a, b, expected }) => {
        expect(updateManager.compareVersions(a, b)).toBe(expected);
      });
    });

    test('should determine correct migrations for version ranges', () => {
      const migrations = updateManager.getMigrationsToRun('1.0.0', '1.2.0');
      
      expect(migrations.length).toBeGreaterThan(0);
      expect(migrations.every(m => m.name && m.execute)).toBe(true);
    });
  });

  describe('Cross-Environment Compatibility', () => {
    test('should work in Firefox environment', async () => {
      // Mock Firefox environment
      delete global.chrome;
      global.browser = {
        runtime: {
          getManifest: jest.fn().mockReturnValue({ version: '1.2.0' }),
          openOptionsPage: jest.fn()
        },
        storage: mockStorage,
        notifications: {
          create: jest.fn(),
          onClicked: { addListener: jest.fn() }
        }
      };

      // Re-import update manager for Firefox
      const { ExtensionUpdateManager } = require('../../js/background.js');
      const firefoxUpdateManager = new ExtensionUpdateManager();

      const installDetails = { reason: 'install' };
      await firefoxUpdateManager.handleInstallation(installDetails);

      expect(global.browser.runtime.openOptionsPage).toHaveBeenCalled();
    });

    test('should handle missing browser APIs gracefully', async () => {
      // Remove notifications API
      delete mockChrome.notifications;

      const installDetails = { reason: 'install' };
      
      // Should not throw even without notifications API
      await expect(updateManager.handleInstallation(installDetails))
        .resolves.not.toThrow();
    });
  });
});

// End-to-end update scenario test
describe('Complete Update Scenario Integration', () => {
  test('should handle real-world update scenario from v1.0.0 to v1.2.0', async () => {
    const mockStorage = {
      sync: { data: {}, get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() },
      local: { data: {}, get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() }
    };

    // Setup realistic storage behavior
    mockStorage.sync.get.mockImplementation(() => Promise.resolve(mockStorage.sync.data));
    mockStorage.sync.set.mockImplementation((items) => {
      Object.assign(mockStorage.sync.data, items);
      return Promise.resolve();
    });
    mockStorage.local.get.mockImplementation(() => Promise.resolve(mockStorage.local.data));
    mockStorage.local.set.mockImplementation((items) => {
      Object.assign(mockStorage.local.data, items);
      return Promise.resolve();
    });

    global.chrome = {
      runtime: { getManifest: jest.fn().mockReturnValue({ version: '1.2.0' }) },
      storage: mockStorage,
      notifications: { create: jest.fn(), onClicked: { addListener: jest.fn() } }
    };

    // Simulate v1.0.0 user data
    mockStorage.sync.data = {
      serverUrl: 'https://user.tldw.com',
      apiKey: 'user-legacy-key', // Old format
      customUserSetting: 'preserve-this',
      extensionVersion: '1.0.0'
    };

    mockStorage.local.data = {
      cache_old_data: 'old-cache',
      userLocalData: 'keep-this'
    };

    const { ExtensionUpdateManager } = require('../../js/background.js');
    const updateManager = new ExtensionUpdateManager();

    // Perform update
    const updateDetails = { reason: 'update', previousVersion: '1.0.0' };
    await updateManager.handleInstallation(updateDetails);

    // Verify final state
    expect(mockStorage.sync.data).toMatchObject({
      serverUrl: 'https://user.tldw.com', // Preserved
      apiToken: 'user-legacy-key', // Migrated from apiKey
      customUserSetting: 'preserve-this', // Preserved
      enableProgressIndicators: true, // New default added
      extensionVersion: '1.2.0', // Updated
      lastUpdated: expect.any(String)
    });

    // Verify legacy key was removed
    expect(mockStorage.sync.data.apiKey).toBeUndefined();

    // Verify backup was created
    const backupKeys = Object.keys(mockStorage.local.data)
      .filter(key => key.startsWith('backup_'));
    expect(backupKeys.length).toBe(1);

    // Verify cache cleanup
    expect(mockStorage.local.data.cache_old_data).toBeUndefined();
    expect(mockStorage.local.data.userLocalData).toBe('keep-this');

    // Verify migration was recorded
    expect(mockStorage.local.data.completedMigrations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: expect.stringContaining('API token'),
          version: '1.2.0'
        })
      ])
    );
  });
});