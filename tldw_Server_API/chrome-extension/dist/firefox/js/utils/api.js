class TLDWApiClient {
  constructor() {
    this.configManager = window.configManager || null;
    this.baseUrl = null;
    this.apiToken = null;
    this.initialized = false;
    this.connectionStatus = {
      isConnected: false,
      lastChecked: null,
      lastSuccessful: null,
      consecutiveFailures: 0
    };
    
    // Request caching and optimization
    this.cache = new Map();
    this.pendingRequests = new Map();
    
    // Listen for configuration changes
    if (this.configManager) {
      this.configManager.addListener((event, data) => {
        if (event === 'updated' && (data.serverUrl || data.apiToken)) {
          this.init();
        }
      });
    }
  }

  async init() {
    try {
      // Use config manager if available, otherwise fallback to storage
      if (this.configManager) {
        await this.configManager.initialize();
        this.baseUrl = this.configManager.getServerUrl();
        this.apiToken = this.configManager.get('apiToken', '');
      } else {
        const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;
        const config = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
        this.baseUrl = config.serverUrl || 'http://localhost:8000';
        this.apiToken = config.apiToken || '';
      }
      
      this.initialized = true;
    } catch (error) {
      console.error('[TLDWApiClient] Initialization failed:', error);
      // Fallback to defaults
      this.baseUrl = 'http://localhost:8000';
      this.apiToken = '';
      this.initialized = true;
    }
  }
  
  getRetryConfig() {
    if (this.configManager) {
      return {
        maxRetries: this.configManager.get('maxRetries', 3),
        baseDelay: this.configManager.get('retryDelay', 1000),
        maxDelay: this.configManager.get('retryDelay', 1000) * Math.pow(this.configManager.get('retryBackoffMultiplier', 2), 3)
      };
    }
    return {
      maxRetries: 3,
      baseDelay: 1000,
      maxDelay: 10000
    };
  }
  
  getCacheConfig() {
    if (this.configManager) {
      return {
        defaultTTL: this.configManager.get('cacheTimeout', 300000),
        maxCacheSize: this.configManager.get('maxCacheSize', 100),
        enabled: this.configManager.get('enableCaching', true),
        cachableEndpoints: ['/prompts/', '/characters/', '/media/']
      };
    }
    return {
      defaultTTL: 300000,
      maxCacheSize: 100,
      enabled: true,
      cachableEndpoints: ['/prompts/', '/characters/', '/media/']
    };
  }
  
  getApiTimeout() {
    return this.configManager ? this.configManager.getApiTimeout() : 30000;
  }

  async checkConnection(withRetry = false) {
    this.connectionStatus.lastChecked = new Date();
    
    try {
      await this.init();
      const timeoutMs = this.configManager ? Math.min(this.getApiTimeout(), 5000) : 5000;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      
      const response = await fetch(`${this.baseUrl}/api/v1/media/`, {
        method: 'GET',
        headers: this.getHeaders(),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      const isConnected = response.ok;
      this.updateConnectionStatus(isConnected);
      return isConnected;
    } catch (error) {
      console.error('Connection check failed:', error);
      this.updateConnectionStatus(false, error);
      
      if (withRetry && this.connectionStatus.consecutiveFailures < this.retryConfig.maxRetries) {
        const delay = this.calculateRetryDelay(this.connectionStatus.consecutiveFailures);
        await this.sleep(delay);
        return this.checkConnection(true);
      }
      
      return false;
    }
  }

  getHeaders(additionalHeaders = {}) {
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Cache-Control': 'no-cache',
      'X-Requested-With': 'XMLHttpRequest',
      ...additionalHeaders
    };
    
    // Add API token if available
    if (this.apiToken) {
      const apiKeyHeader = this.configManager ? 
        this.configManager.get('apiKeyHeader', 'Authorization') : 
        'Authorization';
      headers[apiKeyHeader] = `Bearer ${this.apiToken}`;
    }
    
    // Add security headers
    if (this.configManager && this.configManager.get('enableCORS', true)) {
      headers['Access-Control-Request-Method'] = 'GET, POST, PUT, DELETE, OPTIONS';
      headers['Access-Control-Request-Headers'] = 'Content-Type, Authorization, X-Requested-With';
    }
    
    // Add user agent for identification
    headers['User-Agent'] = this.getUserAgent();
    
    // Add request ID for tracking
    headers['X-Request-ID'] = this.generateRequestId();
    
    return headers;
  }
  
  getUserAgent() {
    const extensionVersion = this.getExtensionVersion();
    const browserInfo = this.getBrowserInfo();
    return `TLDW-Extension/${extensionVersion} (${browserInfo})`;
  }
  
  getExtensionVersion() {
    try {
      const manifest = chrome.runtime.getManifest();
      return manifest.version || '1.0.0';
    } catch (error) {
      return '1.0.0';
    }
  }
  
  getBrowserInfo() {
    const isChrome = typeof chrome !== 'undefined';
    const isFirefox = typeof browser !== 'undefined';
    
    if (isChrome) {
      return `Chrome/${navigator.userAgent.match(/Chrome\/([0-9.]+)/)?.[1] || 'Unknown'}`;
    } else if (isFirefox) {
      return `Firefox/${navigator.userAgent.match(/Firefox\/([0-9.]+)/)?.[1] || 'Unknown'}`;
    } else {
      return 'Unknown Browser';
    }
  }
  
  generateRequestId() {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async request(endpoint, options = {}) {
    if (!this.initialized) {
      await this.init();
    }

    // Use configManager for URL construction if available
    const url = this.configManager ? 
      this.configManager.getApiUrl(endpoint) : 
      `${this.baseUrl}/api/v1${endpoint}`;
      
    const config = {
      ...options,
      headers: this.getHeaders(options.headers || {})
    };
    
    // Add timeout from configuration
    const controller = new AbortController();
    const timeoutMs = this.getApiTimeout();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    config.signal = controller.signal;

    // Check if request is cacheable and use cache if available
    const cacheConfig = this.getCacheConfig();
    const cacheKey = this.generateCacheKey(endpoint, options);
    const method = options.method || 'GET';
    
    if (method === 'GET' && cacheConfig.enabled && this.isCacheable(endpoint)) {
      // Check cache first
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        clearTimeout(timeoutId);
        return cached;
      }
      
      // Check for pending request to avoid duplicate calls
      if (this.pendingRequests.has(cacheKey)) {
        return this.pendingRequests.get(cacheKey);
      }
      
      // Create pending request
      const requestPromise = this.requestWithRetry(url, config, endpoint)
        .then(result => {
          this.setCache(cacheKey, result);
          this.pendingRequests.delete(cacheKey);
          return result;
        })
        .catch(error => {
          this.pendingRequests.delete(cacheKey);
          throw error;
        });
      
      this.pendingRequests.set(cacheKey, requestPromise);
      return requestPromise;
    }
    
    // For non-cacheable requests, invalidate related cache entries
    if (method !== 'GET') {
      this.invalidateCache(endpoint);
    }

    return this.requestWithRetry(url, config, endpoint);
  }

  async requestWithRetry(url, config, endpoint, attempt = 0) {
    try {
      // Handle CORS preflight for complex requests
      if (this.needsPreflightRequest(config)) {
        await this.sendPreflightRequest(url, config);
      }
      
      const response = await fetch(url, config);
      
      // Enhanced error handling for different types of failures
      if (!response.ok) {
        const errorInfo = await this.parseErrorResponse(response);
        const enhancedError = this.createEnhancedError(response.status, errorInfo, endpoint);
        throw enhancedError;
      }

      // Reset connection status on success
      this.updateConnectionStatus(true);

      // Enhanced response parsing with validation
      return await this.parseSuccessResponse(response);
      
    } catch (error) {
      console.error(`API request failed: ${endpoint} (attempt ${attempt + 1})`, error);
      
      // Enhanced error categorization for better retry logic
      const errorCategory = this.categorizeError(error);
      
      // Check if we should retry based on error type
      if (this.shouldRetry(error, attempt, errorCategory)) {
        const delay = this.calculateRetryDelay(attempt);
        console.log(`Retrying ${endpoint} in ${delay}ms... (reason: ${errorCategory})`);
        await this.sleep(delay);
        return this.requestWithRetry(url, config, endpoint, attempt + 1);
      }
      
      this.updateConnectionStatus(false, error);
      throw error;
    }
  }
  
  // CORS and Security Helper Methods
  needsPreflightRequest(config) {
    const method = config.method || 'GET';
    const complexMethods = ['PUT', 'DELETE', 'PATCH'];
    
    // Check if request needs preflight
    if (complexMethods.includes(method.toUpperCase())) {
      return true;
    }
    
    // Check for custom headers that trigger preflight
    const headers = config.headers || {};
    const preflightHeaders = ['Authorization', 'X-API-Key', 'X-Requested-With'];
    
    return preflightHeaders.some(header => 
      Object.keys(headers).some(h => h.toLowerCase() === header.toLowerCase())
    );
  }
  
  async sendPreflightRequest(url, config) {
    try {
      const preflightConfig = {
        method: 'OPTIONS',
        headers: {
          'Access-Control-Request-Method': config.method || 'GET',
          'Access-Control-Request-Headers': Object.keys(config.headers || {}).join(', '),
          'Origin': window.location.origin
        }
      };
      
      const response = await fetch(url, preflightConfig);
      
      if (!response.ok) {
        console.warn('CORS preflight failed:', response.status);
      }
      
      return response;
    } catch (error) {
      console.warn('CORS preflight error:', error);
      // Don't throw here - let the main request proceed
    }
  }
  
  async parseErrorResponse(response) {
    try {
      const contentType = response.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        const text = await response.text();
        return { detail: text || response.statusText };
      }
    } catch (error) {
      return { detail: response.statusText || `HTTP ${response.status}` };
    }
  }
  
  createEnhancedError(status, errorInfo, endpoint) {
    const error = new Error(errorInfo.detail || `HTTP ${status}`);
    error.status = status;
    error.endpoint = endpoint;
    error.timestamp = new Date().toISOString();
    error.requestId = this.generateRequestId();
    
    // Add specific error categories
    if (status === 401) {
      error.category = 'authentication';
      error.userMessage = 'Authentication failed. Please check your API token.';
    } else if (status === 403) {
      error.category = 'authorization';
      error.userMessage = 'Access denied. You may not have permission for this operation.';
    } else if (status === 404) {
      error.category = 'not_found';
      error.userMessage = 'The requested resource was not found.';
    } else if (status === 429) {
      error.category = 'rate_limit';
      error.userMessage = 'Too many requests. Please wait a moment and try again.';
    } else if (status >= 500) {
      error.category = 'server_error';
      error.userMessage = 'Server error. Please try again later.';
    } else if (status === 0) {
      error.category = 'network_error';
      error.userMessage = 'Network error. Please check your connection.';
    } else {
      error.category = 'client_error';
      error.userMessage = errorInfo.detail || 'An unexpected error occurred.';
    }
    
    return error;
  }
  
  async parseSuccessResponse(response) {
    const contentType = response.headers.get('content-type');
    
    try {
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        
        // Validate response structure
        if (this.configManager && this.configManager.isDevelopmentMode()) {
          this.validateResponseStructure(data, response.url);
        }
        
        return data;
      } else {
        return await response.text();
      }
    } catch (error) {
      console.error('Failed to parse response:', error);
      throw new Error('Invalid response format from server');
    }
  }
  
  validateResponseStructure(data, url) {
    // Basic validation for expected response structure
    if (data === null || data === undefined) {
      console.warn(`Empty response from ${url}`);
      return;
    }
    
    // Check for common API response patterns
    if (typeof data === 'object' && !Array.isArray(data)) {
      if (data.error && !data.detail) {
        console.warn(`Non-standard error format from ${url}:`, data);
      }
    }
  }
  
  categorizeError(error) {
    if (error.name === 'AbortError') {
      return 'timeout';
    } else if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
      return 'network';
    } else if (error.message.includes('CORS')) {
      return 'cors';
    } else if (error.status === 429) {
      return 'rate_limit';
    } else if (error.status >= 500) {
      return 'server_error';
    } else if (error.status === 401 || error.status === 403) {
      return 'auth_error';
    } else {
      return 'unknown';
    }
  }
  
  shouldRetry(error, attempt, errorCategory = null) {
    const retryConfig = this.getRetryConfig();
    
    if (attempt >= retryConfig.maxRetries) {
      return false;
    }
    
    // Don't retry certain error types
    const nonRetryableCategories = ['auth_error', 'client_error'];
    if (errorCategory && nonRetryableCategories.includes(errorCategory)) {
      return false;
    }
    
    // Don't retry 4xx errors except for specific cases
    if (error.status >= 400 && error.status < 500) {
      return error.status === 429; // Only retry rate limits
    }
    
    // Retry network errors, timeouts, and 5xx errors
    const retryableCategories = ['timeout', 'network', 'cors', 'rate_limit', 'server_error'];
    return !errorCategory || retryableCategories.includes(errorCategory);
  }

  shouldRetryOld(error, attempt) {
    const retryConfig = this.getRetryConfig();
    if (attempt >= retryConfig.maxRetries) {
      return false;
    }
    
    // Retry on network errors, timeouts, and 5xx status codes
    if (error.name === 'TypeError' || error.message.includes('Failed to fetch')) {
      return true;
    }
    
    if (error.message.includes('HTTP 5')) {
      return true;
    }
    
    return false;
  }

  calculateRetryDelay(attempt) {
    // Exponential backoff with jitter
    const baseDelay = this.retryConfig.baseDelay;
    const exponential = Math.min(baseDelay * Math.pow(2, attempt), this.retryConfig.maxDelay);
    const jitter = Math.random() * 0.3 * exponential;
    return Math.floor(exponential + jitter);
  }

  updateConnectionStatus(isConnected, error = null) {
    const previousStatus = this.connectionStatus.isConnected;
    this.connectionStatus.isConnected = isConnected;
    this.connectionStatus.lastChecked = new Date();
    
    if (isConnected) {
      this.connectionStatus.lastSuccessful = new Date();
      this.connectionStatus.consecutiveFailures = 0;
    } else {
      this.connectionStatus.consecutiveFailures++;
      this.connectionStatus.lastError = error;
    }
    
    // Emit status change event if status changed
    if (previousStatus !== isConnected) {
      this.onConnectionStatusChange?.(isConnected, this.connectionStatus);
    }
  }

  getConnectionStatus() {
    return { ...this.connectionStatus };
  }

  setConnectionStatusCallback(callback) {
    this.onConnectionStatusChange = callback;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Cache management methods
  generateCacheKey(endpoint, options = {}) {
    const method = options.method || 'GET';
    const params = JSON.stringify(options.body || '');
    return `${method}:${endpoint}:${params}`;
  }

  isCacheable(endpoint) {
    return this.cacheConfig.cachableEndpoints.some(cachableEndpoint => 
      endpoint.startsWith(cachableEndpoint)
    );
  }

  getFromCache(key) {
    const cached = this.cache.get(key);
    if (!cached) return null;
    
    const now = Date.now();
    if (now > cached.expiresAt) {
      this.cache.delete(key);
      return null;
    }
    
    return cached.data;
  }

  setCache(key, data, ttl = this.cacheConfig.defaultTTL) {
    // Ensure cache doesn't exceed max size
    if (this.cache.size >= this.cacheConfig.maxCacheSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(key, {
      data: data,
      expiresAt: Date.now() + ttl,
      createdAt: Date.now()
    });
  }

  invalidateCache(endpoint) {
    // Remove all cache entries that start with the endpoint
    for (const key of this.cache.keys()) {
      if (key.includes(endpoint)) {
        this.cache.delete(key);
      }
    }
  }

  clearCache() {
    this.cache.clear();
    this.pendingRequests.clear();
  }

  getCacheStats() {
    const now = Date.now();
    let validEntries = 0;
    let expiredEntries = 0;
    
    for (const [key, value] of this.cache.entries()) {
      if (now > value.expiresAt) {
        expiredEntries++;
      } else {
        validEntries++;
      }
    }
    
    return {
      totalEntries: this.cache.size,
      validEntries,
      expiredEntries,
      pendingRequests: this.pendingRequests.size,
      maxCacheSize: this.cacheConfig.maxCacheSize
    };
  }

  // Chat API
  async createChatCompletion(data) {
    const endpoint = '/chat/completions';
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async streamChatCompletion(data, onChunk) {
    if (!this.initialized) {
      await this.init();
    }

    const url = `${this.baseUrl}/api/v1/chat/completions`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ ...data, stream: true })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          
          try {
            const parsed = JSON.parse(data);
            onChunk(parsed);
          } catch (e) {
            console.error('Failed to parse SSE chunk:', e);
          }
        }
      }
    }
  }

  // Prompts API
  async getPrompts(page = 1, perPage = 10) {
    return this.request(`/prompts/?page=${page}&per_page=${perPage}`);
  }

  async getPrompt(identifier) {
    return this.request(`/prompts/${identifier}`);
  }

  async createPrompt(data) {
    return this.request('/prompts/', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async updatePrompt(identifier, data) {
    return this.request(`/prompts/${identifier}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  async deletePrompt(identifier) {
    return this.request(`/prompts/${identifier}`, {
      method: 'DELETE'
    });
  }

  async searchPrompts(query, page = 1, resultsPerPage = 20) {
    const params = new URLSearchParams({
      search_query: query,
      page: page.toString(),
      results_per_page: resultsPerPage.toString()
    });
    
    return this.request(`/prompts/search?${params}`);
  }

  async exportPrompts(format = 'markdown') {
    return this.request(`/prompts/export?export_format=${format}`);
  }

  // Characters API
  async getCharacters(limit = 100, offset = 0) {
    return this.request(`/characters/?limit=${limit}&offset=${offset}`);
  }

  async getCharacter(id) {
    return this.request(`/characters/${id}`);
  }

  async createCharacter(data) {
    return this.request('/characters/', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async updateCharacter(id, data, expectedVersion) {
    return this.request(`/characters/${id}?expected_version=${expectedVersion}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  async deleteCharacter(id, expectedVersion) {
    return this.request(`/characters/${id}?expected_version=${expectedVersion}`, {
      method: 'DELETE'
    });
  }

  async searchCharacters(query, limit = 10) {
    return this.request(`/characters/search/?query=${encodeURIComponent(query)}&limit=${limit}`);
  }

  async importCharacterCard(file) {
    if (!this.initialized) {
      await this.init();
    }

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/v1/characters/import`, {
      method: 'POST',
      headers: {
        'Token': `Bearer ${this.apiToken}`
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
  }

  // Media API
  async getMediaList(page = 1, resultsPerPage = 10) {
    return this.request(`/media/?page=${page}&results_per_page=${resultsPerPage}`);
  }

  async getMediaItem(id) {
    return this.request(`/media/${id}`);
  }

  async updateMediaItem(id, data) {
    return this.request(`/media/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  async searchMedia(data) {
    return this.request('/media/search', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async processMediaUrl(url, options = {}) {
    return this.request('/media/add', {
      method: 'POST',
      body: JSON.stringify({
        url,
        ...options
      })
    });
  }

  async processMediaFile(file, endpoint, options = {}) {
    if (!this.initialized) {
      await this.init();
    }

    const formData = new FormData();
    formData.append('file', file);
    
    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, value);
    });

    const response = await fetch(`${this.baseUrl}/api/v1/media/${endpoint}`, {
      method: 'POST',
      headers: {
        'Token': `Bearer ${this.apiToken}`
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
  }

  async ingestWebContent(url, options = {}) {
    return this.request('/media/ingest-web-content', {
      method: 'POST',
      body: JSON.stringify({
        url,
        ...options
      })
    });
  }
}

// Create singleton instance
const apiClient = new TLDWApiClient();

// Setup connection status monitoring
apiClient.setConnectionStatusCallback((isConnected, status) => {
  // This will be used by popup.js to update UI
  if (typeof window !== 'undefined' && window.updateConnectionStatus) {
    window.updateConnectionStatus(isConnected, status);
  }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = apiClient;
}