class TLDWApiClient {
  constructor() {
    this.baseUrl = null;
    this.apiToken = null;
    this.initialized = false;
  }

  async init() {
    const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;
    const config = await browserAPI.storage.sync.get(['serverUrl', 'apiToken']);
    this.baseUrl = config.serverUrl || 'http://localhost:8000';
    this.apiToken = config.apiToken || '';
    this.initialized = true;
  }

  async checkConnection() {
    try {
      await this.init();
      const response = await fetch(`${this.baseUrl}/api/v1/media/`, {
        method: 'GET',
        headers: this.getHeaders()
      });
      return response.ok;
    } catch (error) {
      console.error('Connection check failed:', error);
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

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return await response.text();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
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

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = apiClient;
}