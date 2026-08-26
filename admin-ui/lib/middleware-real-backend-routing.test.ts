import { NextRequest } from 'next/server';
import { afterEach, describe, expect, it, vi } from 'vitest';

const originalRealBackendMode = process.env.TLDW_ADMIN_E2E_REAL_BACKEND;
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;

describe('middleware real-backend routing', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
    if (originalRealBackendMode === undefined) {
      delete process.env.TLDW_ADMIN_E2E_REAL_BACKEND;
    } else {
      process.env.TLDW_ADMIN_E2E_REAL_BACKEND = originalRealBackendMode;
    }
    if (originalApiUrl === undefined) {
      delete process.env.NEXT_PUBLIC_API_URL;
    } else {
      process.env.NEXT_PUBLIC_API_URL = originalApiUrl;
    }
  });

  it('validates a single-user API key against the backend mapped from the request port', async () => {
    process.env.TLDW_ADMIN_E2E_REAL_BACKEND = 'true';
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8101';
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const { middleware } = await import('../middleware');
    const request = new NextRequest('http://127.0.0.1:3102/debug', {
      headers: {
        cookie: 'x_api_key=single-user-middleware-routing-key',
      },
    });

    const response = await middleware(request);

    const [validationUrl, validationInit] = fetchMock.mock.calls[0];
    const parsedValidationUrl = new URL(validationUrl);
    expect(parsedValidationUrl.port).toBe('8102');
    expect(parsedValidationUrl.pathname).toBe('/api/v1/users/me');
    expect(validationInit).toEqual(expect.objectContaining({ method: 'GET' }));
    expect(response.headers.get('location')).toBeNull();
  });
});
