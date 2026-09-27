import type { NextRequest } from 'next/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const originalRealBackendMode = process.env.TLDW_ADMIN_E2E_REAL_BACKEND;
const originalSingleUserApiUrl = process.env.TLDW_ADMIN_E2E_SINGLE_USER_API_URL;

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}

vi.mock('next/server', () => ({
  NextResponse: {
    json: (body: unknown, init?: { status?: number; headers?: Record<string, string> }) => ({
      body,
      status: init?.status ?? 200,
      headers: new Map(Object.entries(init?.headers ?? {})),
    }),
  },
}));

const request = { url: 'http://attacker.example:3102/api/health/ready' } as NextRequest;

describe('GET /api/health/ready', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    process.env.TLDW_ADMIN_E2E_REAL_BACKEND = 'true';
    process.env.TLDW_ADMIN_E2E_SINGLE_USER_API_URL = 'http://127.0.0.1:9102';
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    restoreEnv('TLDW_ADMIN_E2E_REAL_BACKEND', originalRealBackendMode);
    restoreEnv('TLDW_ADMIN_E2E_SINGLE_USER_API_URL', originalSingleUserApiUrl);
  });

  it('returns 200 ready when backend is reachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET(request);
    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({ status: 'ready', backend: 'reachable' });
  });

  it('returns 503 not_ready when backend returns non-ok status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 502 }));
    const { GET } = await import('../route');
    const response = await GET(request);
    expect(response.status).toBe(503);
    expect(response.body).toMatchObject({
      status: 'not_ready',
      backend: 'unreachable',
      backend_error: 'Backend returned 502',
    });
  });

  it('returns 503 not_ready when backend is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('ECONNREFUSED')));
    const { GET } = await import('../route');
    const response = await GET(request);
    expect(response.status).toBe(503);
    expect(response.body).toMatchObject({
      status: 'not_ready',
      backend: 'unreachable',
      backend_error: 'Backend unreachable',
    });
  });

  it('returns Cache-Control no-store header', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET(request);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  it('includes timestamp in response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET(request);
    expect(response.body).toHaveProperty('timestamp');
    expect(typeof (response.body as Record<string, unknown>).timestamp).toBe('string');
  });

  it('does not include backend_error when backend is reachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET(request);
    expect(response.body).not.toHaveProperty('backend_error');
  });

  it('uses the configured project backend without trusting the request hostname', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await import('../route');

    await GET(request);

    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:9102/api/v1/health',
      expect.objectContaining({ method: 'GET', cache: 'no-store' }),
    );
  });
});
