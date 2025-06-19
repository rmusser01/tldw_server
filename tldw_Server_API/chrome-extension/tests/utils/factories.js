// Test data factories for consistent test object creation

export const createMockPrompt = (overrides = {}) => ({
  id: Math.random().toString(36).substr(2, 9),
  name: 'Test Prompt',
  details: 'Test prompt details',
  content: 'This is a test prompt content',
  keywords: ['test', 'prompt'],
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides
});

export const createMockCharacter = (overrides = {}) => ({
  id: Math.random().toString(36).substr(2, 9),
  name: 'Test Character',
  description: 'A test character for unit tests',
  personality: 'Helpful and friendly test character',
  instructions: 'You are a test character',
  image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
  created_at: new Date().toISOString(),
  ...overrides
});

export const createMockChatMessage = (overrides = {}) => ({
  role: overrides.role || 'user',
  content: overrides.content || 'Test message content',
  timestamp: overrides.timestamp || new Date().toISOString(),
  ...overrides
});

export const createMockChatResponse = (overrides = {}) => ({
  id: 'chatcmpl-' + Math.random().toString(36).substr(2, 9),
  object: 'chat.completion',
  created: Date.now(),
  model: 'gpt-3.5-turbo',
  choices: [{
    index: 0,
    message: {
      role: 'assistant',
      content: 'This is a test response'
    },
    finish_reason: 'stop'
  }],
  usage: {
    prompt_tokens: 10,
    completion_tokens: 20,
    total_tokens: 30
  },
  ...overrides
});

export const createMockMediaItem = (overrides = {}) => ({
  id: Math.random().toString(36).substr(2, 9),
  title: 'Test Media',
  url: 'https://example.com/media.mp4',
  type: 'video',
  duration: 300,
  size: 1024 * 1024 * 50, // 50MB
  processed: false,
  created_at: new Date().toISOString(),
  ...overrides
});

export const createMockTab = (overrides = {}) => ({
  id: Math.floor(Math.random() * 1000),
  title: 'Test Tab',
  url: 'https://example.com',
  active: false,
  index: 0,
  windowId: 1,
  ...overrides
});

export const createMockConfig = (overrides = {}) => ({
  serverUrl: 'http://localhost:8000',
  apiVersion: 'v1',
  apiKey: null,
  enableStreaming: true,
  theme: 'light',
  defaultModel: 'gpt-3.5-turbo',
  debug: false,
  ...overrides
});

export const createMockError = (message = 'Test error', code = 'TEST_ERROR') => ({
  error: {
    message,
    code,
    details: {}
  }
});

export const createMockStreamChunk = (content, done = false) => ({
  id: 'chatcmpl-' + Math.random().toString(36).substr(2, 9),
  object: 'chat.completion.chunk',
  created: Date.now(),
  model: 'gpt-3.5-turbo',
  choices: [{
    index: 0,
    delta: done ? {} : { content },
    finish_reason: done ? 'stop' : null
  }]
});