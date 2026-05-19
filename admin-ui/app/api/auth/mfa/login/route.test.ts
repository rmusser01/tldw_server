/* @vitest-environment node */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextRequest } from 'next/server';

const buildApiUrlForRequest = vi.fn((_req: unknown, path: string) => `https://example.test${path}`);
const setJwtSessionCookies = vi.fn();
const checkRateLimit = vi.fn(() => ({ allowed: true, retryAfterSeconds: 0 }));
const extractClientIp = vi.fn(() => '127.0.0.1');

vi.mock('@/lib/api-config', () => ({
  buildApiUrlForRequest,
}));

vi.mock('@/lib/server-auth', () => ({
  setJwtSessionCookies,
}));

vi.mock('@/lib/rate-limiter', () => ({
  checkRateLimit,
  extractClientIp,
}));

describe('POST /api/auth/mfa/login', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('returns 504 when the upstream MFA login request times out', async () => {
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
    const responsePromise = POST(new NextRequest('http://localhost/api/auth/mfa/login', {
      method: 'POST',
      body: JSON.stringify({ session_token: 'session-token', otp_code: '123456' }),
      headers: {
        'Content-Type': 'application/json',
      },
    }));

    await vi.advanceTimersByTimeAsync(10_000);

    const response = await responsePromise;
    expect(response.status).toBe(504);
    await expect(response.json()).resolves.toEqual({
      detail: 'MFA login request timed out',
    });
    expect(setJwtSessionCookies).not.toHaveBeenCalled();
  });
});
