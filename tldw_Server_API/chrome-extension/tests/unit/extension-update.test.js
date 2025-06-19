// Unit tests for Extension Update Management
describe('ExtensionUpdateManager', () => {
  let updateManager;
  let mockChrome;
  let mockStorage;

  beforeEach(() => {
    // Mock Chrome APIs
    mockStorage = {
      sync: {
        get: jest.fn(),
        set: jest.fn(),
        clear: jest.fn(),
        remove: jest.fn()
      },
      local: {
        get: jest.fn(),
        set: jest.fn(),
        clear: jest.fn(),
        remove: jest.fn()
      }
    };

    mockChrome = {
      runtime: {
        getManifest: jest.fn().mockReturnValue({ version: '1.2.0' }),
        openOptionsPage: jest.fn()
      },
      storage: mockStorage,
      notifications: {
        create: jest.fn()
      }
    };

    global.chrome = mockChrome;

    // Import update manager class
    const backgroundScript = require('../../js/background.js');
    const { ExtensionUpdateManager } = backgroundScript;
    updateManager = new ExtensionUpdateManager();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Installation Handling', () => {
    test('should handle first install correctly', async () => {
      const installDetails = { reason: 'install' };
      
      mockStorage.sync.get.mockResolvedValue({});
      mockStorage.local.get.mockResolvedValue({});

      await updateManager.handleInstallation(installDetails);

      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: 'http://localhost:8000',
          enableSmartContext: true,
          extensionVersion: '1.2.0',
          firstRun: true
        })
      );

      expect(mockChrome.runtime.openOptionsPage).toHaveBeenCalled();
      expect(mockChrome.notifications.create).toHaveBeenCalledWith(
        'welcome',
        expect.objectContaining({
          title: 'TLDW Extension Installed'
        })
      );
    });

    test('should handle extension update correctly', async () => {
      const updateDetails = { reason: 'update', previousVersion: '1.1.0' };
      
      mockStorage.sync.get.mockResolvedValue({
        serverUrl: 'https://custom.server.com',
        apiToken: 'existing-token'
      });
      mockStorage.local.get.mockResolvedValue({});

      await updateManager.handleInstallation(updateDetails);

      expect(mockStorage.local.set).toHaveBeenCalledWith(
        expect.objectContaining({
          migrationVersion: '1.2.0'
        })
      );

      expect(mockChrome.notifications.create).toHaveBeenCalledWith(
        'update',
        expect.objectContaining({
          title: 'TLDW Extension Updated',
          message: expect.stringContaining('v1.1.0 to v1.2.0')
        })
      );
    });

    test('should handle Chrome update correctly', async () => {
      const chromeUpdateDetails = { reason: 'chrome_update' };
      
      await updateManager.handleInstallation(chromeUpdateDetails);

      expect(mockStorage.local.set).toHaveBeenCalledWith({
        lastConnectionCheck: null,
        connectionStatus: 'unknown'
      });
    });
  });

  describe('Configuration Management', () => {
    test('should set default configuration only for missing keys', async () => {
      const existingConfig = {
        serverUrl: 'https://existing.server.com',
        apiToken: 'existing-token'
      };
      
      mockStorage.sync.get.mockResolvedValue(existingConfig);

      await updateManager.setDefaultConfiguration();

      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.not.objectContaining({
          serverUrl: 'http://localhost:8000' // Should not override existing
        })
      );

      expect(mockStorage.sync.set).toHaveBeenCalledWith(
        expect.objectContaining({
          enableSmartContext: true, // Should add missing defaults
          extensionVersion: '1.2.0'
        })
      );
    });

    test('should initialize extension state correctly', async () => {
      await updateManager.initializeExtensionState();

      expect(mockStorage.local.clear).toHaveBeenCalled();
      expect(mockStorage.local.set).toHaveBeenCalledWith({
        lastConnectionCheck: null,
        connectionStatus: 'unknown',
        cacheInitialized: false,
        migrationVersion: '1.2.0'
      });
    });
  });

  describe('Data Migration', () => {
    test('should run appropriate migrations based on version', async () => {
      const migrations = updateManager.getMigrationsToRun('1.0.0', '1.2.0');
      
      expect(migrations.length).toBeGreaterThan(0);
      expect(migrations[0]).toHaveProperty('name');
      expect(migrations[0]).toHaveProperty('execute');
    });

    test('should execute migrations in order', async () => {
      const executionOrder = [];
      
      // Mock migration strategies
      updateManager.migrationStrategies.set('1.0.0->1.1.0', {
        name: 'Migration 1',
        execute: async () => { executionOrder.push('Migration 1'); }
      });
      
      updateManager.migrationStrategies.set('1.1.0->1.2.0', {
        name: 'Migration 2', 
        execute: async () => { executionOrder.push('Migration 2'); }
      });

      mockStorage.local.get.mockResolvedValue({ completedMigrations: [] });

      await updateManager.migrateData('1.0.0', '1.2.0');

      expect(executionOrder).toEqual(['Migration 1', 'Migration 2']);
    });

    test('should record completed migrations', async () => {
      const migration = {
        name: 'Test Migration',
        execute: jest.fn().mockResolvedValue()
      };

      mockStorage.local.get.mockResolvedValue({ completedMigrations: [] });

      await updateManager.recordMigration(migration);

      expect(mockStorage.local.set).toHaveBeenCalledWith({
        completedMigrations: expect.arrayContaining([
          expect.objectContaining({
            name: 'Test Migration',
            version: '1.2.0',
            timestamp: expect.any(String)
          })
        ])
      });
    });
  });

  describe('Version Comparison', () => {
    test('should compare versions correctly', () => {
      expect(updateManager.compareVersions('1.0.0', '1.0.0')).toBe(0);
      expect(updateManager.compareVersions('1.1.0', '1.0.0')).toBe(1);
      expect(updateManager.compareVersions('1.0.0', '1.1.0')).toBe(-1);
      expect(updateManager.compareVersions('2.0.0', '1.9.9')).toBe(1);
      expect(updateManager.compareVersions('1.0.10', '1.0.2')).toBe(1);
    });

    test('should handle different version formats', () => {
      expect(updateManager.compareVersions('1.0', '1.0.0')).toBe(0);
      expect(updateManager.compareVersions('1', '1.0.0')).toBe(0);
      expect(updateManager.compareVersions('1.2', '1.2.1')).toBe(-1);
    });
  });

  describe('Backup and Recovery', () => {
    test('should create backup before migration', async () => {
      const syncData = { serverUrl: 'test.com', apiToken: 'token' };
      const localData = { cache: 'data' };
      
      mockStorage.sync.get.mockResolvedValue(syncData);
      mockStorage.local.get.mockResolvedValue(localData);

      await updateManager.backupExtensionData();

      expect(mockStorage.local.set).toHaveBeenCalledWith(
        expect.objectContaining({
          [expect.stringMatching(/^backup_\d+$/)]: {
            version: '1.2.0',
            timestamp: expect.any(String),
            syncData,
            localData
          }
        })
      );
    });

    test('should cleanup old backups', async () => {
      const oldBackups = {
        'backup_1000': {},
        'backup_2000': {},
        'backup_3000': {},
        'backup_4000': {}, // This should be kept
        'backup_5000': {}, // This should be kept  
        'backup_6000': {}, // This should be kept
        'other_data': 'keep'
      };

      mockStorage.local.get.mockResolvedValue(oldBackups);

      await updateManager.cleanupOldBackups();

      expect(mockStorage.local.remove).toHaveBeenCalledWith(['backup_1000', 'backup_2000', 'backup_3000']);
    });

    test('should rollback on error', async () => {
      const backupData = {
        backup_1234567890: {
          syncData: { serverUrl: 'backup.com' },
          localData: { cache: 'backup' }
        }
      };

      mockStorage.local.get.mockResolvedValue(backupData);

      await updateManager.rollbackOnError();

      expect(mockStorage.sync.clear).toHaveBeenCalled();
      expect(mockStorage.sync.set).toHaveBeenCalledWith({
        serverUrl: 'backup.com'
      });
    });
  });

  describe('Compatibility Checking', () => {
    test('should detect Chrome version correctly', () => {
      global.navigator = {
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      };

      const chromeVersion = updateManager.getChromeVersion();
      expect(chromeVersion).toBe('91.0.4472.124');
    });

    test('should handle unknown browser gracefully', () => {
      global.navigator = {
        userAgent: 'Unknown Browser'
      };

      const chromeVersion = updateManager.getChromeVersion();
      expect(chromeVersion).toBe('unknown');
    });

    test('should verify compatibility with manifest requirements', async () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      mockChrome.runtime.getManifest.mockReturnValue({
        version: '1.2.0',
        minimum_chrome_version: '95.0'
      });

      global.navigator = {
        userAgent: 'Chrome/90.0.4472.124'
      };

      await updateManager.verifyCompatibility();

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Chrome version 90.0.4472.124 is below minimum required 95.0')
      );

      consoleSpy.mockRestore();
    });
  });

  describe('Data Cleanup', () => {
    test('should cleanup old cache entries', async () => {
      const localData = {
        'cache_old1': 'data',
        'temp_old2': 'data',
        'cache_old3': 'data',
        'keep_this': 'data',
        'normal_data': 'data'
      };

      mockStorage.local.get.mockResolvedValue(localData);

      await updateManager.cleanupCache();

      expect(mockStorage.local.remove).toHaveBeenCalledWith([
        'cache_old1', 'temp_old2', 'cache_old3'
      ]);
    });

    test('should cleanup migration temporary data', async () => {
      await updateManager.cleanupOldData();

      expect(mockStorage.local.remove).toHaveBeenCalledWith([
        'tempMigrationData', 'migrationInProgress'
      ]);
    });
  });

  describe('Error Handling', () => {
    test('should handle migration errors gracefully', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      const failingMigration = {
        name: 'Failing Migration',
        execute: jest.fn().mockRejectedValue(new Error('Migration failed'))
      };

      updateManager.migrationStrategies.set('1.0.0->1.1.0', failingMigration);
      mockStorage.local.get.mockResolvedValue({ completedMigrations: [] });

      await expect(updateManager.migrateData('1.0.0', '1.1.0'))
        .rejects.toThrow('Migration failed');

      expect(consoleSpy).toHaveBeenCalledWith(
        'Migration Failing Migration failed:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });

    test('should handle storage errors in backup creation', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      mockStorage.sync.get.mockRejectedValue(new Error('Storage error'));

      await updateManager.backupExtensionData();

      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to backup extension data:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });
  });
});

// Property-based tests for update manager
describe('ExtensionUpdateManager Property Tests', () => {
  const fc = require('fast-check');
  let updateManager;

  beforeEach(() => {
    global.chrome = {
      runtime: { 
        getManifest: jest.fn().mockReturnValue({ version: '1.0.0' }),
        openOptionsPage: jest.fn()
      },
      storage: {
        sync: { get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() },
        local: { get: jest.fn(), set: jest.fn(), clear: jest.fn(), remove: jest.fn() }
      },
      notifications: { create: jest.fn() }
    };

    const { ExtensionUpdateManager } = require('../../js/background.js');
    updateManager = new ExtensionUpdateManager();
  });

  test('Version comparison should be reflexive', () => {
    fc.assert(fc.property(
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      ([major, minor, patch]) => {
        const version = `${major}.${minor}.${patch}`;
        expect(updateManager.compareVersions(version, version)).toBe(0);
      }
    ));
  });

  test('Version comparison should be antisymmetric', () => {
    fc.assert(fc.property(
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      fc.tuple(fc.nat(20), fc.nat(20), fc.nat(20)),
      ([a1, a2, a3], [b1, b2, b3]) => {
        const versionA = `${a1}.${a2}.${a3}`;
        const versionB = `${b1}.${b2}.${b3}`;
        
        const compAB = updateManager.compareVersions(versionA, versionB);
        const compBA = updateManager.compareVersions(versionB, versionA);
        
        if (compAB > 0) {
          expect(compBA).toBeLessThan(0);
        } else if (compAB < 0) {
          expect(compBA).toBeGreaterThan(0);
        } else {
          expect(compBA).toBe(0);
        }
      }
    ));
  });

  test('Configuration defaults should never override existing values', () => {
    fc.assert(fc.property(
      fc.record({
        serverUrl: fc.webUrl(),
        apiToken: fc.string(),
        enableSmartContext: fc.boolean()
      }),
      async (existingConfig) => {
        chrome.storage.sync.get.mockResolvedValue(existingConfig);
        
        await updateManager.setDefaultConfiguration();
        
        // Verify that existing values are not overridden
        const setCall = chrome.storage.sync.set.mock.calls[0];
        if (setCall) {
          const newConfig = setCall[0];
          
          Object.keys(existingConfig).forEach(key => {
            expect(newConfig).not.toHaveProperty(key);
          });
        }
      }
    ));
  });

  test('Backup cleanup should preserve most recent backups', () => {
    fc.assert(fc.property(
      fc.array(fc.nat(1000000), { minLength: 5, maxLength: 20 }),
      async (timestamps) => {
        const backups = {};
        timestamps.forEach(ts => {
          backups[`backup_${ts}`] = { data: 'test' };
        });
        backups['other_data'] = 'keep';
        
        chrome.storage.local.get.mockResolvedValue(backups);
        
        await updateManager.cleanupOldBackups();
        
        const removeCalls = chrome.storage.local.remove.mock.calls;
        if (removeCalls.length > 0) {
          const removedKeys = removeCalls.flat();
          const backupKeys = Object.keys(backups)
            .filter(key => key.startsWith('backup_'))
            .sort();
          
          // Should only remove if more than 3 backups
          if (backupKeys.length > 3) {
            const shouldKeep = backupKeys.slice(-3);
            const shouldRemove = backupKeys.slice(0, -3);
            
            shouldRemove.forEach(key => {
              expect(removedKeys).toContain(key);
            });
            
            shouldKeep.forEach(key => {
              expect(removedKeys).not.toContain(key);
            });
          }
        }
      }
    ));
  });
});