/** @vitest-environment jsdom */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  addRequestHistory,
  clearRequestHistory,
  getRequestHistory,
} from '../history';

const KEY = 'tldw-request-history';

describe('request history redaction', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('redacts auth-bearing request headers before persisting', () => {
    addRequestHistory({
      id: '1',
      method: 'GET',
      url: '/media/search',
      timestamp: new Date().toISOString(),
      requestHeaders: {
        authorization: 'Bearer XYZ',
        'x-api-key': 'APIKEY-abc',
        'x-csrf-token': 'csrf-123',
        'x-tldw-org-id': 'org-9',
        'content-type': 'application/json',
      },
    });

    const raw = localStorage.getItem(KEY) || '';
    // Distinctive secret values must not appear anywhere in the stored blob.
    expect(raw).not.toContain('Bearer XYZ');
    expect(raw).not.toContain('APIKEY-abc');
    expect(raw).not.toContain('csrf-123');
    expect(raw).not.toContain('org-9');

    const [item] = getRequestHistory();
    expect(item.requestHeaders?.authorization).toBe('[REDACTED]');
    expect(item.requestHeaders?.['x-api-key']).toBe('[REDACTED]');
    expect(item.requestHeaders?.['x-csrf-token']).toBe('[REDACTED]');
    expect(item.requestHeaders?.['x-tldw-org-id']).toBe('[REDACTED]');
    // Non-sensitive headers are preserved for debugging value.
    expect(item.requestHeaders?.['content-type']).toBe('application/json');
  });

  it('matches sensitive header names case-insensitively', () => {
    addRequestHistory({
      id: '1b',
      method: 'GET',
      url: '/media/search',
      timestamp: new Date().toISOString(),
      requestHeaders: {
        Authorization: 'Bearer MIXEDCASE',
        'X-API-KEY': 'MIXED-KEY',
      },
    });

    const raw = localStorage.getItem(KEY) || '';
    expect(raw).not.toContain('MIXEDCASE');
    expect(raw).not.toContain('MIXED-KEY');
  });

  it('does not persist access_token from an /auth/login response body', () => {
    addRequestHistory({
      id: '2',
      method: 'POST',
      url: '/auth/login',
      timestamp: new Date().toISOString(),
      requestHeaders: { authorization: 'Bearer XYZ' },
      responseBody: { access_token: 'SECRET-TOKEN-123', token_type: 'bearer' },
    });

    const raw = localStorage.getItem(KEY) || '';
    expect(raw).not.toContain('SECRET-TOKEN-123');
    expect(raw).not.toContain('Bearer XYZ');
  });

  it('redacts access_token/refresh_token on /auth/refresh responses', () => {
    addRequestHistory({
      id: '2b',
      method: 'POST',
      url: '/auth/refresh',
      timestamp: new Date().toISOString(),
      responseBody: { access_token: 'REFRESH-ACCESS', refresh_token: 'REFRESH-REFRESH' },
    });

    const raw = localStorage.getItem(KEY) || '';
    expect(raw).not.toContain('REFRESH-ACCESS');
    expect(raw).not.toContain('REFRESH-REFRESH');
  });

  it('redacts access_token/refresh_token keys on non-auth routes too', () => {
    addRequestHistory({
      id: '3',
      method: 'GET',
      url: '/some/route',
      timestamp: new Date().toISOString(),
      responseBody: {
        data: { access_token: 'LEAK-1', nested: { refresh_token: 'LEAK-2' } },
        list: [{ access_token: 'LEAK-3' }],
      },
    });

    const raw = localStorage.getItem(KEY) || '';
    expect(raw).not.toContain('LEAK-1');
    expect(raw).not.toContain('LEAK-2');
    expect(raw).not.toContain('LEAK-3');
  });

  it('clearRequestHistory empties the store', () => {
    addRequestHistory({
      id: '4',
      method: 'GET',
      url: '/x',
      timestamp: new Date().toISOString(),
    });
    expect(getRequestHistory().length).toBe(1);

    clearRequestHistory();

    expect(getRequestHistory()).toEqual([]);
    expect(localStorage.getItem(KEY)).toBeNull();
  });
});
