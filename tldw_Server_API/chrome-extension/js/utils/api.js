class TLDWApiClient {
  constructor() {
    this.baseUrl = null;
    this.apiToken = null;
    this.initialized = false;
    this.retryConfig = {
      maxRetries: 3,
      baseDelay: 1000,
      maxDelay: 10000
    };
    this.connectionStatus = {
      isConnected: false,
      lastChecked: null,
      lastSuccessful: null,
      consecutiveFailures: 0
    };
    
    // Request caching and optimization
    this.cache = new Map();
    this.pendingRequests = new Map();
    this.cacheConfig = {
      defaultTTL: 5 * 60 * 1000, // 5 minutes
      maxCacheSize: 100,
      cachableEndpoints: [
        '/prompts/',
        '/characters/',
        '/media/'
      ]
    };
  }

  async init() {
    const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;
    const config = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
    this.baseUrl = config.serverUrl || 'http://localhost:8000';
    this.apiToken = config.apiToken || '';
    this.initialized = true;
  }

  async checkConnection(withRetry = false) {
    this.connectionStatus.lastChecked = new Date();
    
    try {
      await this.init();
      const response = await fetch(`${this.baseUrl}/api/v1/media/`, {
        method: 'GET',
        headers: this.getHeaders(),
        timeout: 5000
      });
      
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
      ...additionalHeaders
    };
    
    if (this.apiToken) {
      headers['Token'] = `Bearer ${this.apiToken}`;
    }
    
    return headers;
  }

  async request(endpoint, options = {}) {
    if (!this.initialized) {
      await this.init();
    }

    const url = `${this.baseUrl}/api/v1${endpoint}`;
    const config = {
      ...options,
      headers: this.getHeaders(options.headers || {})
    };

    // Check if request is cacheable and use cache if available
    const cacheKey = this.generateCacheKey(endpoint, options);
    const method = options.method || 'GET';
    
    if (method === 'GET' && this.isCacheable(endpoint)) {
      // Check cache first
      const cached = this.getFromCache(cacheKey);
      if (cached) {
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
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      // Reset connection status on success
      this.updateConnectionStatus(true);

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return await response.text();
    } catch (error) {
      console.error(`API request failed: ${endpoint} (attempt ${attempt + 1})`, error);
      
      // Check if we should retry
      if (this.shouldRetry(error, attempt)) {
        const delay = this.calculateRetryDelay(attempt);
        console.log(`Retrying ${endpoint} in ${delay}ms...`);
        await this.sleep(delay);
        return this.requestWithRetry(url, config, endpoint, attempt + 1);
      }
      
      this.updateConnectionStatus(false, error);
      throw error;
    }
  }

  shouldRetry(error, attempt) {
    if (attempt >= this.retryConfig.maxRetries) {
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