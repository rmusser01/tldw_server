/**
 * Integration tests for API interactions
 * These tests verify the complete flow of API calls through the extension
 */

import '../../js/utils/api.js';

describe('API Integration Tests', () => {
  let apiClient;
  const baseUrl = 'http://localhost:8000';
  const apiToken = 'test-integration-token';

  beforeEach(() => {
    apiClient = new TLDWApiClient();
    
    // Mock storage to return test configuration
    chrome.storage.sync.get.mockResolvedValue({
      serverUrl: baseUrl,
      apiToken: apiToken
    });
  });

  describe('Chat API Integration', () => {
    it('should complete a full chat conversation flow', async () => {
      // Mock successful chat completion
      fetch.mockResponseOnce(JSON.stringify({
        id: 'chat-123',
        object: 'chat.completion',
        created: Date.now(),
        model: 'gpt-4',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: 'Hello! I can help you with that.'
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18
        },
        conversation_id: 'conv-456'
      }));

      const chatRequest = {
        model: 'gpt-4',
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Hello, can you help me?' }
        ],
        temperature: 0.7,
        max_tokens: 150
      };

      const response = await apiClient.createChatCompletion(chatRequest);

      expect(fetch).toHaveBeenCalledWith(
        `${baseUrl}/api/v1/chat/completions`,
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Token': `Bearer ${apiToken}`
          }),
          body: JSON.stringify(chatRequest)
        })
      );

      expect(response).toHaveProperty('choices');
      expect(response.choices[0].message.content).toBe('Hello! I can help you with that.');
      expect(response.conversation_id).toBe('conv-456');
    });

    it('should handle chat with character context', async () => {
      fetch.mockResponseOnce(JSON.stringify({
        choices: [{
          message: {
            role: 'assistant',
            content: '*speaks in character voice* Greetings, traveler!'
          }
        }]
      }));

      const chatRequest = {
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello' }],
        character_id: 'char-123'
      };

      const response = await apiClient.createChatCompletion(chatRequest);

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"character_id":"char-123"')
        })
      );

      expect(response.choices[0].message.content).toContain('Greetings, traveler!');
    });

    it('should handle streaming chat responses', async () => {
      const streamChunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        'data: {"choices":[{"delta":{"content":" there"}}]}\n',
        'data: {"choices":[{"delta":{"content":"!"}}]}\n',
        'data: [DONE]\n'
      ];

      const mockStream = new ReadableStream({
        start(controller) {
          streamChunks.forEach(chunk => {
            controller.enqueue(new TextEncoder().encode(chunk));
          });
          controller.close();
        }
      });

      fetch.mockResponseOnce('', {
        body: mockStream,
        headers: { 'content-type': 'text/event-stream' }
      });

      const chunks = [];
      await apiClient.streamChatCompletion(
        { model: 'gpt-4', messages: [{ role: 'user', content: 'Hi' }] },
        chunk => chunks.push(chunk)
      );

      expect(chunks).toHaveLength(3);
      expect(chunks.map(c => c.choices[0].delta.content).join('')).toBe('Hello there!');
    });
  });

  describe('Prompts API Integration', () => {
    it('should perform full CRUD operations on prompts', async () => {
      // Create prompt
      const newPrompt = {
        name: 'Test Prompt',
        system_prompt: 'You are a helpful assistant',
        user_prompt: 'Help me with {task}',
        keywords: ['test', 'assistant']
      };

      fetch.mockResponseOnce(JSON.stringify({
        id: 1,
        ...newPrompt,
        created_at: new Date().toISOString()
      }));

      const created = await apiClient.createPrompt(newPrompt);
      expect(created.id).toBe(1);

      // Get prompt
      fetch.mockResponseOnce(JSON.stringify({
        id: 1,
        ...newPrompt,
        version: 1
      }));

      const retrieved = await apiClient.getPrompt(1);
      expect(retrieved.name).toBe('Test Prompt');

      // Update prompt
      const updates = { ...newPrompt, name: 'Updated Prompt' };
      fetch.mockResponseOnce(JSON.stringify({
        id: 1,
        ...updates,
        version: 2
      }));

      const updated = await apiClient.updatePrompt(1, updates);
      expect(updated.name).toBe('Updated Prompt');

      // Delete prompt
      fetch.mockResponseOnce('', { status: 204 });
      await apiClient.deletePrompt(1);

      expect(fetch).toHaveBeenCalledTimes(4);
    });

    it('should search and export prompts', async () => {
      // Search prompts
      fetch.mockResponseOnce(JSON.stringify({
        results: [
          { id: 1, name: 'Code Review', keywords: ['code', 'review'] },
          { id: 2, name: 'Code Documentation', keywords: ['code', 'docs'] }
        ],
        total: 2,
        page: 1
      }));

      const searchResults = await apiClient.searchPrompts('code');
      expect(searchResults.results).toHaveLength(2);
      expect(searchResults.results[0].name).toBe('Code Review');

      // Export prompts
      const exportContent = btoa('# Prompts Export\n\n## Code Review\n...');
      fetch.mockResponseOnce(JSON.stringify({
        content: exportContent,
        filename: 'prompts_export.md',
        format: 'markdown'
      }));

      const exported = await apiClient.exportPrompts('markdown');
      expect(exported.content).toBe(exportContent);
      expect(atob(exported.content)).toContain('# Prompts Export');
    });
  });

  describe('Characters API Integration', () => {
    it('should import and manage character cards', async () => {
      // Import character card
      const characterFile = new File(
        ['mock character data'],
        'character.png',
        { type: 'image/png' }
      );

      fetch.mockResponseOnce(JSON.stringify({
        character: {
          id: 1,
          name: 'Assistant Character',
          description: 'A helpful AI assistant',
          personality: 'Friendly and knowledgeable'
        },
        message: 'Character imported successfully'
      }));

      const imported = await apiClient.importCharacterCard(characterFile);
      expect(imported.character.name).toBe('Assistant Character');

      // Search characters
      fetch.mockResponseOnce(JSON.stringify([
        {
          id: 1,
          name: 'Assistant Character',
          match_score: 0.95
        }
      ]));

      const searchResults = await apiClient.searchCharacters('assistant');
      expect(searchResults).toHaveLength(1);
      expect(searchResults[0].name).toBe('Assistant Character');

      // Update character
      fetch.mockResponseOnce(JSON.stringify({
        id: 1,
        name: 'Updated Assistant',
        version: 2
      }));

      const updated = await apiClient.updateCharacter(
        1,
        { name: 'Updated Assistant' },
        1 // expected version
      );
      expect(updated.name).toBe('Updated Assistant');
    });
  });

  describe('Media API Integration', () => {
    it('should process various media types', async () => {
      // Process video URL
      fetch.mockResponseOnce(JSON.stringify({
        success: true,
        media_id: 1,
        title: 'Example Video',
        transcript: 'This is the video transcript...',
        summary: 'Video summary'
      }));

      const videoResult = await apiClient.processMediaUrl(
        'https://example.com/video.mp4',
        {
          chunk_method: 'tokens',
          chunk_size: 500
        }
      );
      expect(videoResult.media_id).toBe(1);
      expect(videoResult.transcript).toBeDefined();

      // Process PDF file
      const pdfFile = new File(
        ['%PDF-1.4 mock content'],
        'document.pdf',
        { type: 'application/pdf' }
      );

      fetch.mockResponseOnce(JSON.stringify({
        success: true,
        content: 'Extracted PDF content...',
        page_count: 10,
        chunks: ['chunk1', 'chunk2', 'chunk3']
      }));

      const pdfResult = await apiClient.processMediaFile(
        pdfFile,
        'process-pdfs',
        { extract_images: false }
      );
      expect(pdfResult.page_count).toBe(10);
      expect(pdfResult.chunks).toHaveLength(3);

      // Get media list
      fetch.mockResponseOnce(JSON.stringify({
        media_items: [
          {
            id: 1,
            title: 'Example Video',
            type: 'video',
            created_at: new Date().toISOString()
          }
        ],
        total: 1,
        page: 1
      }));

      const mediaList = await apiClient.getMediaList();
      expect(mediaList.media_items).toHaveLength(1);
      expect(mediaList.media_items[0].title).toBe('Example Video');
    });

    it('should handle web content ingestion', async () => {
      fetch.mockResponseOnce(JSON.stringify({
        success: true,
        media_id: 2,
        title: 'Web Article',
        content: 'Article content...',
        metadata: {
          author: 'John Doe',
          published_date: '2024-01-15'
        }
      }));

      const result = await apiClient.ingestWebContent(
        'https://example.com/article',
        {
          extract_media: true,
          include_comments: false
        }
      );

      expect(result.media_id).toBe(2);
      expect(result.metadata.author).toBe('John Doe');
    });
  });

  describe('Error Handling Integration', () => {
    it('should handle authentication errors across all APIs', async () => {
      fetch.mockResponseOnce(
        JSON.stringify({ detail: 'Invalid authentication credentials' }),
        { status: 401 }
      );

      await expect(apiClient.createChatCompletion({
        model: 'gpt-4',
        messages: []
      })).rejects.toThrow('Invalid authentication credentials');

      fetch.mockResponseOnce(
        JSON.stringify({ detail: 'Invalid authentication credentials' }),
        { status: 401 }
      );

      await expect(apiClient.getPrompts()).rejects.toThrow('Invalid authentication credentials');
    });

    it('should handle rate limiting', async () => {
      fetch.mockResponseOnce(
        JSON.stringify({
          detail: 'Rate limit exceeded',
          retry_after: 60
        }),
        {
          status: 429,
          headers: { 'Retry-After': '60' }
        }
      );

      await expect(apiClient.createChatCompletion({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'test' }]
      })).rejects.toThrow('Rate limit exceeded');
    });

    it('should handle network errors gracefully', async () => {
      fetch.mockRejectOnce(new Error('Network failure'));

      await expect(apiClient.getMediaList()).rejects.toThrow('Network failure');

      // Verify connection check also handles network errors
      fetch.mockRejectOnce(new Error('Network failure'));
      const isConnected = await apiClient.checkConnection();
      expect(isConnected).toBe(false);
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle multiple concurrent API calls', async () => {
      // Mock different responses for concurrent calls
      fetch
        .mockResponseOnce(JSON.stringify({ prompts: [], total: 0 }))
        .mockResponseOnce(JSON.stringify([]))
        .mockResponseOnce(JSON.stringify({ media_items: [], total: 0 }));

      const [prompts, characters, media] = await Promise.all([
        apiClient.getPrompts(),
        apiClient.getCharacters(),
        apiClient.getMediaList()
      ]);

      expect(fetch).toHaveBeenCalledTimes(3);
      expect(prompts).toHaveProperty('prompts');
      expect(Array.isArray(characters)).toBe(true);
      expect(media).toHaveProperty('media_items');
    });
  });

  describe('Real-world Scenarios', () => {
    it('should handle a complete user workflow', async () => {
      // User selects text on a webpage and sends to chat
      const selectedText = 'Explain quantum computing in simple terms';
      
      // Create chat with selected text
      fetch.mockResponseOnce(JSON.stringify({
        choices: [{
          message: {
            role: 'assistant',
            content: 'Quantum computing uses quantum bits...'
          }
        }],
        conversation_id: 'conv-789'
      }));

      const chatResponse = await apiClient.createChatCompletion({
        model: 'gpt-4',
        messages: [{ role: 'user', content: selectedText }]
      });

      // Save the response as a prompt
      const promptData = {
        name: 'Quantum Computing Explainer',
        system_prompt: 'Explain complex topics simply',
        user_prompt: selectedText
      };

      fetch.mockResponseOnce(JSON.stringify({
        id: 10,
        ...promptData
      }));

      const savedPrompt = await apiClient.createPrompt(promptData);

      // Process related webpage as media
      fetch.mockResponseOnce(JSON.stringify({
        success: true,
        media_id: 5,
        title: 'Quantum Computing Article'
      }));

      const mediaResult = await apiClient.ingestWebContent(
        'https://example.com/quantum-article'
      );

      // Verify the complete workflow
      expect(chatResponse.conversation_id).toBe('conv-789');
      expect(savedPrompt.id).toBe(10);
      expect(mediaResult.media_id).toBe(5);
    });
  });
});