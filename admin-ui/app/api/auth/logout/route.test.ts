/* @vitest-environment node */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextRequest } from 'next/server';

const buildApiUrlForRequest = vi.fn((_req: unknown, path: string) => `https://example.test${path}`);
const getBackendAuthHeaders = vi.fn(() => new Headers({ Authorization: 'Bearer test-token' }));
const clearAdminSessionCookies = vi.fn();
const invalidateAuthCache = vi.fn().mockResolvedValue(undefined);
const loggerWarn = vi.fn();

vi.mock('@/lib/api-config', () => ({
  buildApiUrlForRequest,
}));

vi.mock('@/lib/server-auth', () => ({
  clearAdminSessionCookies,
  getBackendAuthHeaders,
  ACCESS_TOKEN_COOKIE: 'access_token',
  API_KEY_COOKIE: 'x_api_key',
  LEGACY_API_KEY_COOKIE: 'x-api-key',
}));

vi.mock('@/middleware', () => ({
  invalidateAuthCache,
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    warn: loggerWarn,
  },
}));

describe('POST /api/auth/logout', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it('logs backend logout failures and still clears local session cookies', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error('backend unavailable'));
    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('./route');
    const request = new NextRequest('http://localhost/api/auth/logout', {
      method: 'POST',
      headers: { cookie: 'access_token=test-jwt-token' },
    });
    const response = await POST(request);

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });
    expect(getBackendAuthHeaders).toHaveBeenCalledWith(request);
    expect(fetchMock).toHaveBeenCalledWith('https://example.test/auth/logout', {
      method: 'POST',
      headers: new Headers({ Authorization: 'Bearer test-token' }),
      cache: 'no-store',
    });
    expect(loggerWarn).toHaveBeenCalledTimes(1);
    expect(loggerWarn).toHaveBeenCalledWith(
      'Backend logout failed',
      expect.objectContaining({
        component: 'auth/logout',
        error: 'backend unavailable',
      }),
    );
    expect(clearAdminSessionCookies).toHaveBeenCalledTimes(1);
    expect(clearAdminSessionCookies).toHaveBeenCalledWith(expect.objectContaining({
      cookies: expect.anything(),
    }));
    expect(invalidateAuthCache).toHaveBeenCalled();
  });
});
