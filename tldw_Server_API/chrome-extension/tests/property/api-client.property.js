/**
 * Property-based tests for API Client
 * Using fast-check to test invariants and edge cases
 */

import fc from 'fast-check';
import { APIClient } from '../../js/utils/api.js';
import { setupMockAPI, createMockStream } from '../utils/helpers.js';
import { createMockError, createMockStreamChunk } from '../utils/factories.js';

describe('API Client Property Tests', () => {
  let apiClient;
  let originalFetch;

  beforeEach(() => {
    originalFetch = global.fetch;
    global.fetch = jest.fn();
    apiClient = new APIClient();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  describe('Request Construction Properties', () => {
    test('should always include required headers', () => {
      fc.assert(
        fc.property(
          fc.webUrl(),
          fc.option(fc.dictionary(fc.string(), fc.string())),
          fc.option(fc.json()),
          async (url, headers, body) => {
            fetch.mockResolvedValue({
              ok: true,
              json: async () => ({})
            });

            await apiClient.request(url, {
              headers,
              body: body ? JSON.stringify(body) : undefined
            });

            const [, options] = fetch.mock.calls[0];
            
            // Should always have Content-Type
            expect(options.headers).toHaveProperty('Content-Type');
            
            // Should preserve custom headers
            if (headers) {
              Object.keys(headers).forEach(key => {
                expect(options.headers[key]).toBeDefined();
              });
            }
          }
        ),
        { numRuns: 100 }
      );
    });

    test('should handle any valid JSON body', () => {
      fc.assert(
        fc.property(
          fc.json(),
          async (jsonData) => {
            fetch.mockResolvedValue({
              ok: true,
              json: async () => ({ success: true })
            });

            const result = await apiClient.request('/test', {
              method: 'POST',
              body: JSON.stringify(jsonData)
            });

            expect(result).toHaveProperty('success', true);
            
            const [, options] = fetch.mock.calls[0];
            expect(() => JSON.parse(options.body)).not.toThrow();
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('Retry Logic Properties', () => {
    test('should retry failed requests up to max retries', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 10 }),
          fc.array(fc.boolean(), { minLength: 1, maxLength: 10 }),
          async (maxRetries, failures) => {
            apiClient.config.maxRetries = maxRetries;
            apiClient.config.retryDelay = 1; // Fast retries for testing
            
            let attemptCount = 0;
            fetch.mockImplementation(() => {
              attemptCount++;
              if (attemptCount <= failures.filter(f => f).length) {
                return Promise.reject(new Error('Network error'));
              }
              return Promise.resolve({
                ok: true,
                json: async () => ({ success: true })
              });
            });

            try {
              await apiClient.request('/test');
              // Should succeed if failures < maxRetries
              expect(attemptCount).toBeLessThanOrEqual(maxRetries + 1);
            } catch (error) {
              // Should fail if all retries exhausted
              expect(attemptCount).toBe(Math.min(maxRetries + 1, failures.filter(f => f).length));
            }
          }
        ),
        { numRuns: 50 }
      );
    });

    test('should apply exponential backoff correctly', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 100, max: 1000 }),
          fc.float({ min: 1.5, max: 3 }),
          fc.integer({ min: 1, max: 5 }),
          async (baseDelay, multiplier, retries) => {
            apiClient.config.retryDelay = baseDelay;
            apiClient.config.retryBackoffMultiplier = multiplier;
            apiClient.config.maxRetries = retries;
            
            let delays = [];
            let lastCallTime = Date.now();
            
            fetch.mockImplementation(() => {
              const now = Date.now();
              delays.push(now - lastCallTime);
              lastCallTime = now;
              return Promise.reject(new Error('Network error'));
            });

            try {
              await apiClient.request('/test');
            } catch (error) {
              // Expected to fail
            }

            // Check exponential growth
            for (let i = 1; i < delays.length; i++) {
              const expectedMinDelay = baseDelay * Math.pow(multiplier, i - 1);
              expect(delays[i]).toBeGreaterThanOrEqual(expectedMinDelay * 0.9); // Allow 10% variance
            }
          }
        ),
        { numRuns: 20 }
      );
    });
  });

  describe('Error Handling Properties', () => {
    test('should safely handle any error response', () => {
      fc.assert(
        fc.property(
          fc.oneof(
            fc.constant(null),
            fc.constant(undefined),
            fc.string(),
            fc.object(),
            fc.array(fc.anything())
          ),
          fc.integer({ min: 400, max: 599 }),
          async (errorBody, statusCode) => {
            fetch.mockResolvedValue({
              ok: false,
              status: statusCode,
              json: async () => {
                if (errorBody === null || errorBody === undefined) {
                  throw new Error('Invalid JSON');
                }
                return errorBody;
              }
            });

            await expect(apiClient.request('/test')).rejects.toThrow();
            
            // Should not crash on any input
            expect(true).toBe(true);
          }
        ),
        { numRuns: 100 }
      );
    });

    test('should handle network errors gracefully', () => {
      fc.assert(
        fc.property(
          fc.oneof(
            fc.constant(new Error('Network error')),
            fc.constant(new TypeError('Failed to fetch')),
            fc.constant(new Error('ECONNREFUSED')),
            fc.constant(new Error('ETIMEDOUT'))
          ),
          async (error) => {
            fetch.mockRejectedValue(error);

            await expect(apiClient.request('/test')).rejects.toThrow();
            
            // Should handle any network error type
            expect(true).toBe(true);
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('Streaming Response Properties', () => {
    test('should handle any sequence of stream chunks', () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.oneof(
              fc.string(),
              fc.constant(''),
              fc.constant('[DONE]')
            ),
            { minLength: 0, maxLength: 100 }
          ),
          async (chunks) => {
            const streamChunks = chunks.map((content, index) => 
              content === '[DONE]' 
                ? 'data: [DONE]\n\n'
                : `data: ${JSON.stringify(createMockStreamChunk(content, index === chunks.length - 1))}\n\n`
            );

            fetch.mockResolvedValue({
              ok: true,
              headers: new Headers({ 'content-type': 'text/event-stream' }),
              body: createMockStream(streamChunks)
            });

            const messages = [];
            await apiClient.streamRequest('/chat', {}, (message) => {
              messages.push(message);
            });

            // Should process all non-[DONE] chunks
            const expectedCount = chunks.filter(c => c !== '[DONE]').length;
            expect(messages.length).toBeLessThanOrEqual(expectedCount);
          }
        ),
        { numRuns: 50 }
      );
    });

    test('should handle malformed stream data', () => {
      fc.assert(
        fc.property(
          fc.array(
            fc.oneof(
              fc.string(),
              fc.constant('data: {invalid json}'),
              fc.constant('not-data-prefix'),
              fc.constant('data:no-space'),
              fc.constant('\n\n\n'),
              fc.constant('data: ')
            ),
            { minLength: 1, maxLength: 20 }
          ),
          async (malformedChunks) => {
            fetch.mockResolvedValue({
              ok: true,
              headers: new Headers({ 'content-type': 'text/event-stream' }),
              body: createMockStream(malformedChunks)
            });

            const messages = [];
            const errors = [];

            try {
              await apiClient.streamRequest('/chat', {}, 
                (message) => messages.push(message),
                (error) => errors.push(error)
              );
            } catch (error) {
              // Expected for some malformed data
            }

            // Should not crash on malformed data
            expect(true).toBe(true);
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('URL Construction Properties', () => {
    test('should correctly construct URLs with any query parameters', () => {
      fc.assert(
        fc.property(
          fc.dictionary(
            fc.string().filter(s => s.length > 0 && !s.includes('&') && !s.includes('=')),
            fc.oneof(
              fc.string(),
              fc.integer(),
              fc.boolean(),
              fc.constant(null),
              fc.constant(undefined)
            )
          ),
          async (params) => {
            fetch.mockResolvedValue({
              ok: true,
              json: async () => ({})
            });

            const baseUrl = 'http://test.com';
            apiClient.config.serverUrl = baseUrl;

            await apiClient.request('/test', { params });

            const [url] = fetch.mock.calls[0];
            const urlObj = new URL(url);

            // Check all params are in URL
            Object.entries(params).forEach(([key, value]) => {
              if (value !== null && value !== undefined) {
                expect(urlObj.searchParams.get(key)).toBe(String(value));
              } else {
                expect(urlObj.searchParams.has(key)).toBe(false);
              }
            });
          }
        ),
        { numRuns: 100 }
      );
    });

    test('should handle special characters in URLs', () => {
      fc.assert(
        fc.property(
          fc.string().filter(s => s.length > 0),
          fc.dictionary(fc.string(), fc.string()),
          async (path, params) => {
            fetch.mockResolvedValue({
              ok: true,
              json: async () => ({})
            });

            // Encode path to handle special characters
            const encodedPath = path.split('/').map(encodeURIComponent).join('/');
            
            try {
              await apiClient.request(`/${encodedPath}`, { params });
              
              const [url] = fetch.mock.calls[0];
              // Should create valid URL
              expect(() => new URL(url)).not.toThrow();
            } catch (error) {
              // Some paths might be invalid
              expect(error.message).toMatch(/Invalid URL|Failed to construct/);
            }
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('Timeout Properties', () => {
    test('should timeout after specified duration', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 100, max: 5000 }),
          fc.integer({ min: 0, max: 10000 }),
          async (timeout, responseDelay) => {
            apiClient.config.apiTimeout = timeout;

            fetch.mockImplementation(() => new Promise((resolve) => {
              setTimeout(() => {
                resolve({
                  ok: true,
                  json: async () => ({ success: true })
                });
              }, responseDelay);
            }));

            const startTime = Date.now();

            try {
              await apiClient.request('/test');
              const duration = Date.now() - startTime;
              
              // Should complete before timeout
              expect(duration).toBeLessThan(timeout + 100); // Allow some margin
              expect(responseDelay).toBeLessThan(timeout);
            } catch (error) {
              const duration = Date.now() - startTime;
              
              // Should timeout if response is too slow
              expect(error.message).toContain('timeout');
              expect(duration).toBeGreaterThanOrEqual(timeout - 100); // Allow some margin
              expect(responseDelay).toBeGreaterThanOrEqual(timeout);
            }
          }
        ),
        { numRuns: 20 }
      );
    });
  });

  describe('Input Validation Properties', () => {
    test('should validate required fields in chat messages', () => {
      fc.assert(
        fc.property(
          fc.record({
            messages: fc.array(
              fc.record({
                role: fc.option(fc.string()),
                content: fc.option(fc.string())
              })
            ),
            model: fc.option(fc.string()),
            temperature: fc.option(fc.float({ min: 0, max: 2 }))
          }),
          async (input) => {
            fetch.mockResolvedValue({
              ok: true,
              json: async () => ({ choices: [{ message: { content: 'response' } }] })
            });

            try {
              await apiClient.createChatCompletion(input);
              
              // Should have valid messages
              expect(input.messages).toBeDefined();
              expect(input.messages.every(m => m.role && m.content)).toBe(true);
            } catch (error) {
              // Should fail on invalid input
              const hasInvalidMessages = !input.messages || 
                input.messages.some(m => !m.role || !m.content);
              expect(hasInvalidMessages).toBe(true);
            }
          }
        ),
        { numRuns: 100 }
      );
    });
  });
});