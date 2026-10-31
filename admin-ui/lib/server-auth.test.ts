/* @vitest-environment node */

import { describe, expect, it } from 'vitest';
import type { NextRequest } from 'next/server';

import {
  ACCESS_TOKEN_COOKIE,
  API_KEY_COOKIE,
  appendProxyHeaders,
  buildProxyResponse,
  getBackendAuthHeaders,
} from './server-auth';

const requestStub = (
  headers: HeadersInit = {},
  cookies: Record<string, string> = {},
): NextRequest => ({
  headers: new Headers(headers),
  cookies: {
    get: (name: string) => {
      const value = cookies[name];
      return value === undefined ? undefined : { name, value };
    },
  },
} as unknown as NextRequest);

describe('server-side proxy header boundaries', () => {
  it('forwards only the explicit conditional and idempotency command headers', () => {
    const request = requestStub({
      authorization: 'Bearer browser-override',
      'x-api-key': 'browser-api-key',
      'if-match': '"admin-webhook-41-r1"',
      'idempotency-key': '0123456789abcdef',
      'x-injected': 'drop-me',
    });
    const headers = new Headers({ authorization: 'Bearer cookie-token' });

    appendProxyHeaders(request, headers);

    expect(headers.get('authorization')).toBe('Bearer cookie-token');
    expect(headers.get('x-api-key')).toBeNull();
    expect(headers.get('if-match')).toBe('"admin-webhook-41-r1"');
    expect(headers.get('idempotency-key')).toBe('0123456789abcdef');
    expect(headers.get('x-injected')).toBeNull();
    expect(headers.get('x-request-id')).toBeTruthy();
  });

  it('keeps cookie-derived authorization authoritative over request headers', () => {
    const jwtRequest = requestStub(
      { 'x-api-key': 'browser-api-key' },
      {
        [ACCESS_TOKEN_COOKIE]: 'cookie-jwt',
        [API_KEY_COOKIE]: 'cookie-api-key',
      },
    );
    const apiKeyRequest = requestStub(
      { 'x-api-key': 'browser-api-key' },
      { [API_KEY_COOKIE]: 'cookie-api-key' },
    );

    expect(Object.fromEntries(getBackendAuthHeaders(jwtRequest))).toEqual({
      authorization: 'Bearer cookie-jwt',
    });
    expect(Object.fromEntries(getBackendAuthHeaders(apiKeyRequest))).toEqual({
      'x-api-key': 'cookie-api-key',
    });
  });

  it('preserves bounded webhook response metadata while dropping transport headers', async () => {
    const response = new Response('{"ok":true}', {
      status: 200,
      headers: {
        'content-type': 'application/json',
        etag: '"admin-webhook-41-r2"',
        'cache-control': 'no-store',
        pragma: 'no-cache',
        'x-request-id': 'request-41',
        'content-length': '11',
        'set-cookie': 'do-not-forward=1',
      },
    });

    const proxied = await buildProxyResponse(response);

    expect(proxied.headers.get('etag')).toBe('"admin-webhook-41-r2"');
    expect(proxied.headers.get('cache-control')).toBe('no-store');
    expect(proxied.headers.get('pragma')).toBe('no-cache');
    expect(proxied.headers.get('x-request-id')).toBe('request-41');
    expect(proxied.headers.get('content-length')).toBeNull();
    expect(proxied.headers.get('set-cookie')).toBeNull();
  });
});
