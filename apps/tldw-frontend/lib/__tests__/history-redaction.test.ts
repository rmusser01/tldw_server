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

  it('omits auth-bearing request headers before persisting', () => {
    addRequestHistory({
      id: '1',
      method: 'GET',
      url: '/media/search',
      timestamp: new Date().toISOString(),
      requestHeaders: {
        authorization: 'Bearer XYZ',
        cookie: 'session=COOKIE-123',
        'proxy-authorization': 'Basic PROXY-456',
        'set-cookie': 'refresh=SET-COOKIE-789',
        'x-api-key': 'APIKEY-abc',
        'x-auth-token': 'AUTH-TOKEN-def',
        'x-csrf-token': 'csrf-123',
        'x-tldw-org-id': 'org-9',
        'content-type': 'application/json',
      },
    });

    const raw = localStorage.getItem(KEY) || '';
    // Distinctive secret values must not appear anywhere in the stored blob.
    expect(raw).not.toContain('Bearer XYZ');
    expect(raw).not.toContain('COOKIE-123');
    expect(raw).not.toContain('PROXY-456');
    expect(raw).not.toContain('SET-COOKIE-789');
    expect(raw).not.toContain('APIKEY-abc');
    expect(raw).not.toContain('AUTH-TOKEN-def');
    expect(raw).not.toContain('csrf-123');
    expect(raw).not.toContain('org-9');

    const [item] = getRequestHistory();
    expect(item.requestHeaders).not.toHaveProperty('authorization');
    expect(item.requestHeaders).not.toHaveProperty('cookie');
    expect(item.requestHeaders).not.toHaveProperty('proxy-authorization');
    expect(item.requestHeaders).not.toHaveProperty('set-cookie');
    expect(item.requestHeaders).not.toHaveProperty('x-api-key');
    expect(item.requestHeaders).not.toHaveProperty('x-auth-token');
    expect(item.requestHeaders).not.toHaveProperty('x-csrf-token');
    expect(item.requestHeaders).not.toHaveProperty('x-tldw-org-id');
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

  it('redacts additional credential-shaped body keys on non-auth routes', () => {
    addRequestHistory({
      id: '3b',
      method: 'POST',
      url: '/some/route',
      timestamp: new Date().toISOString(),
      requestBody: {
        id_token: 'IDT-LEAK',
        session_token: 'SESS-LEAK',
        api_key: 'APIK-LEAK',
        apiKey: 'APIK2-LEAK',
        'x-api-key': 'XAPIK-LEAK',
        jwt: 'JWT-LEAK',
        secret: 'SEC-LEAK',
        password: 'PW-LEAK',
        client_secret: 'CS-LEAK',
        keep: 'visible-value',
      },
      responseBody: { data: { api_key: 'RESP-APIK-LEAK' } },
    });

    const raw = localStorage.getItem(KEY) || '';
    for (const leaked of [
      'IDT-LEAK',
      'SESS-LEAK',
      'APIK-LEAK',
      'APIK2-LEAK',
      'XAPIK-LEAK',
      'JWT-LEAK',
      'SEC-LEAK',
      'PW-LEAK',
      'CS-LEAK',
      'RESP-APIK-LEAK',
    ]) {
      expect(raw).not.toContain(leaked);
    }
    // Non-sensitive keys stay readable for debugging value.
    expect(raw).toContain('visible-value');

    const [item] = getRequestHistory();
    const body = item.requestBody as Record<string, unknown>;
    expect(body.id_token).toBe('[REDACTED]');
    expect(body.api_key).toBe('[REDACTED]');
    expect(body.apiKey).toBe('[REDACTED]');
    expect(body.client_secret).toBe('[REDACTED]');
    expect(body.keep).toBe('visible-value');
  });

  it('fails closed on tokens nested deeper than the redaction depth limit', () => {
    // Bury a token below the (>6) recursion depth. The old fail-open behavior
    // returned the raw subtree past the limit, leaking the token.
    addRequestHistory({
      id: '3c',
      method: 'POST',
      url: '/some/route',
      timestamp: new Date().toISOString(),
      responseBody: {
        a: { b: { c: { d: { e: { f: { g: { access_token: 'DEEP-LEAK' } } } } } } },
      },
    });

    const raw = localStorage.getItem(KEY) || '';
    expect(raw).not.toContain('DEEP-LEAK');
    // The truncated subtree is replaced with the redaction placeholder.
    expect(raw).toContain('[REDACTED]');
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
