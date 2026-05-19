/* @vitest-environment node */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextRequest } from 'next/server';

const setApiKeySessionCookies = vi.fn();
const buildApiUrlForRequest = vi.fn((_req: unknown, path: string) => `https://example.test${path}`);
const checkRateLimit = vi.fn(() => ({ allowed: true, retryAfterSeconds: 0 }));
const extractClientIp = vi.fn(() => '127.0.0.1');

vi.mock('@/lib/api-config', () => ({
  buildApiUrlForRequest,
}));

vi.mock('@/lib/server-auth', () => ({
  setApiKeySessionCookies,
}));

vi.mock('@/lib/rate-limiter', () => ({
  checkRateLimit,
  extractClientIp,
}));

describe('POST /api/auth/apikey', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.useRealTimers();
    process.env.AUTH_MODE = 'single_user';
    process.env.ADMIN_UI_ALLOW_API_KEY_LOGIN = 'true';
    process.env.NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN = 'false';
    delete process.env.ADMIN_UI_ENTERPRISE_MODE;
  });

  it('rejects API-key login when enterprise mode is enabled', async () => {
    process.env.ADMIN_UI_ENTERPRISE_MODE = 'true';
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('./route');
    const response = await POST(new NextRequest('http://localhost/api/auth/apikey', {
      method: 'POST',
      body: JSON.stringify({ apiKey: 'admin-key' }),
      headers: {
        'Content-Type': 'application/json',
      },
    }));

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({
      detail: 'Admin UI API key login is disabled. Use multi-user credentials.',
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(setApiKeySessionCookies).not.toHaveBeenCalled();
  });

  it('rejects API-key login when the admin UI is not running in single-user auth mode', async () => {
    process.env.AUTH_MODE = 'multi_user';
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('./route');
    const response = await POST(new NextRequest('http://localhost/api/auth/apikey', {
      method: 'POST',
      body: JSON.stringify({ apiKey: 'admin-key' }),
      headers: {
        'Content-Type': 'application/json',
      },
    }));

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({
      detail: 'Admin UI API key login is disabled. Use multi-user credentials.',
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(setApiKeySessionCookies).not.toHaveBeenCalled();
  });

  it('rejects API-key login when only the public UI flag is enabled', async () => {
    process.env.ADMIN_UI_ALLOW_API_KEY_LOGIN = 'false';
    process.env.NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN = 'true';
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('./route');
    const response = await POST(new NextRequest('http://localhost/api/auth/apikey', {
      method: 'POST',
      body: JSON.stringify({ apiKey: 'admin-key' }),
      headers: {
        'Content-Type': 'application/json',
      },
    }));

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({
      detail: 'Admin UI API key login is disabled. Use multi-user credentials.',
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(setApiKeySessionCookies).not.toHaveBeenCalled();
  });

  it('returns 504 when API-key validation times out', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
      if (!init?.signal) {
        return Promise.reject(new Error('missing signal'));
      }

      return new Promise((_resolve, reject) => {
        const abort = () => reject(new DOMException('The operation was aborted.', 'AbortError'));
        if (init.signal?.aborted) {
          abort();
          return;
        }
        init.signal?.addEventListener('abort', abort, { once: true });
      });
    }));

    const { POST } = await import('./route');
    const responsePromise = POST(new NextRequest('http://localhost/api/auth/apikey', {
      method: 'POST',
      body: JSON.stringify({ apiKey: 'admin-key' }),
      headers: {
        'Content-Type': 'application/json',
      },
    }));

    await vi.advanceTimersByTimeAsync(10_000);

    const response = await responsePromise;
    expect(response.status).toBe(504);
    await expect(response.json()).resolves.toEqual({
      detail: 'API key validation timed out',
    });
    expect(setApiKeySessionCookies).not.toHaveBeenCalled();
  });
});
