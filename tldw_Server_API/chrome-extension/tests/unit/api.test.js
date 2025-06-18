import '../../js/utils/api.js';

describe('TLDWApiClient', () => {
  let apiClient;

  beforeEach(() => {
    // Create a new instance for each test
    apiClient = new TLDWApiClient();
  });

  describe('init', () => {
    it('should initialize with default values', async () => {
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000',
        apiToken: 'test-token'
      });

      await apiClient.init();

      expect(apiClient.baseUrl).toBe('http://localhost:8000');
      expect(apiClient.apiToken).toBe('test-token');
      expect(apiClient.initialized).toBe(true);
    });

    it('should use empty token if not provided', async () => {
      chrome.storage.sync.get.mockResolvedValue({
        serverUrl: 'http://localhost:8000'
      });

      await apiClient.init();

      expect(apiClient.apiToken).toBe('');
    });
  });

  describe('checkConnection', () => {
    it('should return true when connection is successful', async () => {
      fetch.mockResponseOnce(JSON.stringify({ success: true }), { status: 200 });

      const result = await apiClient.checkConnection();

      expect(result).toBe(true);
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/media/',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Token': 'Bearer test-token'
          })
        })
      );
    });

    it('should return false when connection fails', async () => {
      fetch.mockRejectOnce(new Error('Network error'));

      const result = await apiClient.checkConnection();

      expect(result).toBe(false);
    });
  });

  describe('getHeaders', () => {
    beforeEach(async () => {
      await apiClient.init();
    });

    it('should return headers with Bearer token', () => {
      const headers = apiClient.getHeaders();

      expect(headers).toEqual({
        'Content-Type': 'application/json',
        'Token': 'Bearer test-token'
      });
    });

    it('should merge additional headers', () => {
      const headers = apiClient.getHeaders({ 'X-Custom': 'value' });

      expect(headers).toEqual({
        'Content-Type': 'application/json',
        'Token': 'Bearer test-token',
        'X-Custom': 'value'
      });
    });

    it('should not include token header if apiToken is empty', async () => {
      apiClient.apiToken = '';
      const headers = apiClient.getHeaders();

      expect(headers).toEqual({
        'Content-Type': 'application/json'
      });
    });
  });

  describe('request', () => {
    beforeEach(async () => {
      await apiClient.init();
    });

    it('should make successful GET request', async () => {
      const mockResponse = { data: 'test' };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.request('/test');

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/test',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Token': 'Bearer test-token'
          })
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should make successful POST request with body', async () => {
      const requestData = { name: 'test' };
      const mockResponse = { id: 1, name: 'test' };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.request('/test', {
        method: 'POST',
        body: JSON.stringify(requestData)
      });

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/test',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(requestData),
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should handle non-JSON responses', async () => {
      fetch.mockResponseOnce('Plain text response', {
        headers: { 'content-type': 'text/plain' }
      });

      const result = await apiClient.request('/test');

      expect(result).toBe('Plain text response');
    });

    it('should throw error on failed request', async () => {
      fetch.mockResponseOnce(
        JSON.stringify({ detail: 'Not found' }),
        { status: 404 }
      );

      await expect(apiClient.request('/test')).rejects.toThrow('Not found');
    });

    it('should handle network errors', async () => {
      fetch.mockRejectOnce(new Error('Network failure'));

      await expect(apiClient.request('/test')).rejects.toThrow('Network failure');
    });
  });

  describe('Chat API methods', () => {
    beforeEach(async () => {
      await apiClient.init();
    });

    it('should create chat completion', async () => {
      const chatData = {
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello' }]
      };
      const mockResponse = {
        choices: [{ message: { content: 'Hi there!' } }]
      };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.createChatCompletion(chatData);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/chat/completions',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(chatData)
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should stream chat completion', async () => {
      const chatData = {
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello' }]
      };
      
      const mockChunks = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
        'data: {"choices":[{"delta":{"content":" there!"}}]}\n',
        'data: [DONE]\n'
      ];

      const mockReadableStream = new ReadableStream({
        start(controller) {
          mockChunks.forEach(chunk => {
            controller.enqueue(new TextEncoder().encode(chunk));
          });
          controller.close();
        }
      });

      fetch.mockResponseOnce('', {
        body: mockReadableStream,
        headers: { 'content-type': 'text/event-stream' }
      });

      const chunks = [];
      const onChunk = jest.fn(chunk => chunks.push(chunk));

      await apiClient.streamChatCompletion(chatData, onChunk);

      expect(onChunk).toHaveBeenCalledTimes(2);
      expect(chunks[0]).toEqual({ choices: [{ delta: { content: 'Hi' } }] });
      expect(chunks[1]).toEqual({ choices: [{ delta: { content: ' there!' } }] });
    });
  });

  describe('Prompts API methods', () => {
    beforeEach(async () => {
      await apiClient.init();
    });

    it('should get prompts with pagination', async () => {
      const mockResponse = {
        prompts: [{ id: 1, name: 'Test Prompt' }],
        total: 1
      };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.getPrompts(1, 20);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/prompts/?page=1&per_page=20',
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse);
    });

    it('should search prompts', async () => {
      const mockResponse = {
        results: [{ id: 1, name: 'Test Prompt' }],
        total: 1
      };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.searchPrompts('test', 1, 10);

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('search_query=test'),
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse);
    });

    it('should export prompts', async () => {
      const mockResponse = {
        content: 'base64content',
        filename: 'prompts.md'
      };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.exportPrompts('markdown');

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/prompts/export?export_format=markdown',
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Characters API methods', () => {
    beforeEach(async () => {
      await apiClient.init();
    });

    it('should get characters', async () => {
      const mockResponse = [
        { id: 1, name: 'Character 1' },
        { id: 2, name: 'Character 2' }
      ];
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.getCharacters(50, 0);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/characters/?limit=50&offset=0',
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse);
    });

    it('should import character card', async () => {
      const mockFile = new File(['content'], 'character.png', { type: 'image/png' });
      const mockResponse = {
        character: { id: 1, name: 'Imported Character' }
      };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.importCharacterCard(mockFile);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/characters/import',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Media API methods', () => {
    beforeEach(async () => {
      await apiClient.init();
    });

    it('should get media list', async () => {
      const mockResponse = {
        media_items: [{ id: 1, title: 'Test Media' }],
        total: 1
      };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.getMediaList(1, 10);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/media/?page=1&results_per_page=10',
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse);
    });

    it('should process media URL', async () => {
      const mockResponse = { success: true, media_id: 1 };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.processMediaUrl('https://example.com/video', {
        title: 'Test Video'
      });

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/media/add',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            url: 'https://example.com/video',
            title: 'Test Video'
          })
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should process media file', async () => {
      const mockFile = new File(['content'], 'video.mp4', { type: 'video/mp4' });
      const mockResponse = { success: true, transcript: 'Test transcript' };
      fetch.mockResponseOnce(JSON.stringify(mockResponse));

      const result = await apiClient.processMediaFile(mockFile, 'process-videos', {
        chunk_size: 300
      });

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/media/process-videos',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });
});