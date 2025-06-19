// Configuration Management System for TLDW Browser Extension
// Centralizes all configuration values and provides environment-specific settings

class ConfigManager {
  constructor() {
    this.defaultConfig = {
      // Server Configuration
      serverUrl: 'http://localhost:8000',
      apiVersion: 'v1',
      apiTimeout: 30000,
      
      // Connection Settings
      connectionCheckInterval: 30000,
      maxRetries: 3,
      retryDelay: 1000,
      retryBackoffMultiplier: 2,
      
      // Cache Settings
      cacheTimeout: 300000, // 5 minutes
      maxCacheSize: 100,
      enableCaching: true,
      
      // UI Settings
      toastDuration: 4000,
      progressUpdateInterval: 100,
      debounceDelay: 300,
      
      // Search Settings
      searchMinLength: 2,
      maxRecentSearches: 10,
      searchCacheTimeout: 60000, // 1 minute
      
      // Batch Processing
      batchDelay: 1000,
      batchConcurrency: 3,
      maxBatchSize: 50,
      
      // Security Settings
      enableCORS: true,
      allowedOrigins: ['http://localhost:8000', 'https://localhost:8443'],
      apiKeyHeader: 'X-API-Key',
      
      // Feature Flags
      enableSmartContext: true,
      enableBatchOperations: true,
      enableProgressIndicators: true,
      enableAdvancedSearch: true,
      enableFloatingButton: true,
      
      // File Upload Settings
      maxFileSize: 100 * 1024 * 1024, // 100MB
      supportedFormats: ['mp4', 'mp3', 'pdf', 'epub', 'doc', 'docx', 'txt'],
      uploadChunkSize: 1024 * 1024, // 1MB chunks
      
      // Development Settings
      debug: false,
      logLevel: 'info', // 'debug', 'info', 'warn', 'error'
      enablePerformanceMetrics: false
    };
    
    this.environments = {
      development: {
        serverUrl: 'http://localhost:8000',
        debug: true,
        logLevel: 'debug',
        enablePerformanceMetrics: true
      },
      staging: {
        serverUrl: 'https://staging.tldw.example.com',
        debug: false,
        logLevel: 'info'
      },
      production: {
        serverUrl: 'https://api.tldw.example.com',
        debug: false,
        logLevel: 'warn',
        enableCaching: true,
        cacheTimeout: 600000 // 10 minutes in production
      }
    };
    
    this.currentConfig = null;
    this.listeners = new Set();
  }
  
  async initialize() {
    try {
      // Load user settings from storage
      const userSettings = await this.loadUserSettings();
      
      // Detect environment
      const environment = this.detectEnvironment();
      
      // Merge configurations: defaults -> environment -> user settings
      this.currentConfig = {
        ...this.defaultConfig,
        ...this.environments[environment],
        ...userSettings,
        environment
      };
      
      // Validate configuration
      this.validateConfig();
      
      // Notify listeners
      this.notifyListeners('initialized', this.currentConfig);
      
      console.log(`[ConfigManager] Initialized with environment: ${environment}`, this.currentConfig);
      
    } catch (error) {
      console.error('[ConfigManager] Failed to initialize:', error);
      this.currentConfig = { ...this.defaultConfig, environment: 'development' };
    }
  }
  
  async loadUserSettings() {
    if (typeof chrome !== 'undefined' && chrome.storage) {
      return new Promise((resolve) => {
        chrome.storage.sync.get([
          'serverUrl', 'apiTimeout', 'enableCaching', 'debug', 'logLevel',
          'toastDuration', 'maxFileSize', 'enableSmartContext', 'enableBatchOperations'
        ], (result) => {
          resolve(result || {});
        });
      });
    } else if (typeof browser !== 'undefined' && browser.storage) {
      return browser.storage.sync.get();
    } else {
      // Fallback to localStorage for testing
      const stored = localStorage.getItem('tldw-config');
      return stored ? JSON.parse(stored) : {};
    }
  }
  
  async saveUserSettings(settings) {
    const settingsToSave = { ...settings };
    
    try {
      if (typeof chrome !== 'undefined' && chrome.storage) {
        chrome.storage.sync.set(settingsToSave);
      } else if (typeof browser !== 'undefined' && browser.storage) {
        await browser.storage.sync.set(settingsToSave);
      } else {
        localStorage.setItem('tldw-config', JSON.stringify(settingsToSave));
      }
      
      // Update current config
      this.currentConfig = { ...this.currentConfig, ...settingsToSave };
      
      // Notify listeners
      this.notifyListeners('updated', settingsToSave);
      
    } catch (error) {
      console.error('[ConfigManager] Failed to save settings:', error);
      throw error;
    }
  }
  
  detectEnvironment() {
    // Check for development indicators
    if (this.isDevelopment()) {
      return 'development';
    }
    
    // Check for staging indicators
    if (this.isStaging()) {
      return 'staging';
    }
    
    // Default to production
    return 'production';
  }
  
  isDevelopment() {
    // Check various development indicators
    return (
      location.hostname === 'localhost' ||
      location.hostname === '127.0.0.1' ||
      location.hostname.startsWith('dev.') ||
      (typeof process !== 'undefined' && process?.env?.NODE_ENV === 'development') ||
      this.defaultConfig.debug === true
    );
  }
  
  isStaging() {
    return (
      location.hostname.includes('staging') ||
      location.hostname.includes('test') ||
      (typeof process !== 'undefined' && process?.env?.NODE_ENV === 'staging')
    );
  }
  
  validateConfig() {
    const config = this.currentConfig;
    
    // Validate server URL
    if (!config.serverUrl || !this.isValidUrl(config.serverUrl)) {
      console.warn('[ConfigManager] Invalid server URL, using default');
      config.serverUrl = this.defaultConfig.serverUrl;
    }
    
    // Validate numeric values
    const numericFields = ['apiTimeout', 'connectionCheckInterval', 'maxRetries', 'cacheTimeout'];
    numericFields.forEach(field => {
      if (typeof config[field] !== 'number' || config[field] <= 0) {
        console.warn(`[ConfigManager] Invalid ${field}, using default`);
        config[field] = this.defaultConfig[field];
      }
    });
    
    // Validate arrays
    if (!Array.isArray(config.allowedOrigins)) {
      config.allowedOrigins = this.defaultConfig.allowedOrigins;
    }
    
    // Ensure server URL is in allowed origins for CORS
    if (!config.allowedOrigins.includes(config.serverUrl)) {
      config.allowedOrigins.push(config.serverUrl);
    }
  }
  
  isValidUrl(string) {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  }
  
  get(key, defaultValue = null) {
    if (!this.currentConfig) {
      console.warn('[ConfigManager] Not initialized, using default config');
      return this.defaultConfig[key] ?? defaultValue;
    }
    
    return this.currentConfig[key] ?? defaultValue;
  }
  
  set(key, value) {
    if (!this.currentConfig) {
      console.warn('[ConfigManager] Not initialized');
      return;
    }
    
    this.currentConfig[key] = value;
    
    // Save to persistent storage if it's a user setting
    const userSettings = { [key]: value };
    this.saveUserSettings(userSettings);
  }
  
  getAll() {
    return this.currentConfig ? { ...this.currentConfig } : { ...this.defaultConfig };
  }
  
  // Specific getters for commonly used values
  getServerUrl() {
    return this.get('serverUrl');
  }
  
  getApiUrl(endpoint = '') {
    const baseUrl = this.getServerUrl();
    const apiVersion = this.get('apiVersion');
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${baseUrl}/api/${apiVersion}${cleanEndpoint}`;
  }
  
  getApiTimeout() {
    return this.get('apiTimeout');
  }
  
  isFeatureEnabled(feature) {
    return this.get(feature, false);
  }
  
  isDevelopmentMode() {
    return this.get('environment') === 'development' || this.get('debug', false);
  }
  
  // Configuration presets for quick setup
  applyPreset(presetName) {
    const presets = {
      performance: {
        enableCaching: true,
        cacheTimeout: 600000,
        debounceDelay: 150,
        batchConcurrency: 5
      },
      security: {
        enableCORS: true,
        apiTimeout: 10000,
        maxRetries: 2
      },
      development: {
        debug: true,
        logLevel: 'debug',
        enablePerformanceMetrics: true,
        toastDuration: 8000
      },
      minimal: {
        enableSmartContext: false,
        enableBatchOperations: false,
        enableAdvancedSearch: false,
        enableFloatingButton: false
      }
    };
    
    const preset = presets[presetName];
    if (preset) {
      Object.assign(this.currentConfig, preset);
      this.saveUserSettings(preset);
      this.notifyListeners('preset_applied', { preset: presetName, config: preset });
    }
  }
  
  // Event system for configuration changes
  addListener(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
  
  notifyListeners(event, data) {
    this.listeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('[ConfigManager] Listener error:', error);
      }
    });
  }
  
  // Export/Import configuration
  exportConfig() {
    const exportData = {
      version: '1.0',
      timestamp: new Date().toISOString(),
      config: this.getAll(),
      environment: this.get('environment')
    };
    
    return JSON.stringify(exportData, null, 2);
  }
  
  async importConfig(configJson) {
    try {
      const importData = JSON.parse(configJson);
      
      if (!importData.config) {
        throw new Error('Invalid configuration format');
      }
      
      // Validate imported config
      const validatedConfig = { ...this.defaultConfig, ...importData.config };
      this.currentConfig = validatedConfig;
      this.validateConfig();
      
      // Save to storage
      await this.saveUserSettings(validatedConfig);
      
      this.notifyListeners('imported', validatedConfig);
      
      return true;
    } catch (error) {
      console.error('[ConfigManager] Failed to import config:', error);
      throw error;
    }
  }
  
  // Reset to defaults
  async resetToDefaults() {
    this.currentConfig = { ...this.defaultConfig };
    await this.saveUserSettings({});
    this.notifyListeners('reset', this.currentConfig);
  }
  
  // Configuration health check
  healthCheck() {
    const issues = [];
    const config = this.currentConfig;
    
    if (!this.isValidUrl(config.serverUrl)) {
      issues.push('Invalid server URL');
    }
    
    if (config.apiTimeout < 1000) {
      issues.push('API timeout too low (minimum 1000ms recommended)');
    }
    
    if (config.maxFileSize > 500 * 1024 * 1024) {
      issues.push('Max file size too large (500MB maximum recommended)');
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      config: this.getAll()
    };
  }
}

// Global configuration instance
const configManager = new ConfigManager();

// Browser API compatibility for configuration
const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;

// Auto-initialize when the script loads
if (typeof window !== 'undefined') {
  configManager.initialize();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ConfigManager, configManager };
} else if (typeof window !== 'undefined') {
  window.configManager = configManager;
}