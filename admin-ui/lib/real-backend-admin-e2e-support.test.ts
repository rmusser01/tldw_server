import { afterEach, describe, expect, it, vi } from 'vitest';

import { postAdminE2EJson } from '@/tests/e2e/real-backend/helpers/admin-e2e-support';

const baseUrl = 'http://127.0.0.1:8101';
const supportPath = '/api/v1/test-support/admin-e2e';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('real-backend admin support requests', () => {
  it.each([
    ['/seed', { scenario: 'jwt_admin' }],
    ['/reset', undefined],
  ])('returns the server-issued CSRF token on %s', async (path, body) => {
    vi.stubEnv('TLDW_ADMIN_E2E_SUPPORT_KEY', 'configured-support-key');
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === `${baseUrl}/health` && !init?.method) {
        return Response.json({ status: 'ok' }, {
          headers: {
            'X-CSRF-Token': 'server-issued.unbound',
            'Set-Cookie': 'csrf_token=server-issued.unbound; Path=/; SameSite=lax',
          },
        });
      }
      const headers = new Headers(init?.headers);
      if (url === `${baseUrl}${supportPath}${path}`
        && init?.method === 'POST'
        && init.body === (body === undefined ? undefined : JSON.stringify(body))
        && headers.get('X-TLDW-Admin-E2E-Key') === 'configured-support-key'
        && headers.get('Cookie') === 'csrf_token=server-issued.unbound'
        && headers.get('X-CSRF-Token') === 'server-issued.unbound') {
        return Response.json({ status: 'ok' });
      }
      return Response.json({ detail: 'CSRF token validation failed' }, { status: 403 });
    }));

    await expect(postAdminE2EJson(baseUrl, `${supportPath}${path}`, body))
      .resolves.toEqual({ status: 'ok' });
  });

  it('retains the support endpoint error when CSRF is disabled', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => (
      url === `${baseUrl}/health`
        ? Response.json({ status: 'ok' })
        : Response.json({ detail: 'Invalid support key' }, { status: 403 })
    )));

    await expect(postAdminE2EJson(baseUrl, `${supportPath}/reset`)).rejects.toThrow(
      `Admin e2e request failed for ${supportPath}/reset: 403 {"detail":"Invalid support key"}`,
    );
  });

  it('stops before a support write and preserves failed bootstrap evidence', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('backend unavailable', { status: 503 }));
    vi.stubGlobal('fetch', fetch);

    await expect(postAdminE2EJson(baseUrl, `${supportPath}/reset`)).rejects.toThrow(
      'Admin e2e CSRF bootstrap failed: 503 backend unavailable',
    );
    expect(fetch).toHaveBeenCalledExactlyOnceWith(`${baseUrl}/health`);
  });
});
