import { describe, expect, it } from 'vitest';
import { isServicePromptRequestPath } from '../service-prompt-scope-error';

describe('saved Chat title captured request scope', () => {
  it.each(['/api/v1/chats/owned', '/api/v1/chats/owned?expected_version=7'])(
    'permits exact PUT %s',
    (path) => {
      expect(isServicePromptRequestPath(path, 'PUT')).toBe(true);
    }
  );
  it.each([
    ['PUT', '/api/v1/chats/'],
    ['POST', '/api/v1/chats/owned'],
    ['PATCH', '/api/v1/chats/owned'],
    ['PUT', '/api/v1/chats/owned/messages'],
    ['PUT', '/api/v1/chats/owned/nested'],
    ['PUT', '/api/v1/chats/a%2fb'],
    ['PUT', '/api/v1/chats/%2e%2e'],
    ['PUT', 'https://foreign.test/api/v1/chats/owned'],
  ])('rejects expanded or malformed %s %s', (method, path) => {
    expect(isServicePromptRequestPath(path, method)).toBe(false);
  });
});
