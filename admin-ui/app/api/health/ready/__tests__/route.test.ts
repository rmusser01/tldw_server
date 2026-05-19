import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('next/server', () => ({
  NextResponse: {
    json: (body: unknown, init?: { status?: number; headers?: Record<string, string> }) => ({
      body,
      status: init?.status ?? 200,
      headers: new Map(Object.entries(init?.headers ?? {})),
    }),
  },
}));

vi.mock('@/lib/api-config', () => ({
  buildApiUrl: (path: string) => `http://localhost:8000/api/v1${path}`,
}));

describe('GET /api/health/ready', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  it('returns 200 ready when backend is reachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET();
    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({ status: 'ready', backend: 'reachable' });
  });

  it('returns 503 not_ready when backend returns non-ok status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 502 }));
    const { GET } = await import('../route');
    const response = await GET();
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
    const response = await GET();
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
    const response = await GET();
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  it('includes timestamp in response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET();
    expect(response.body).toHaveProperty('timestamp');
    expect(typeof (response.body as Record<string, unknown>).timestamp).toBe('string');
  });

  it('does not include backend_error when backend is reachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');
    const response = await GET();
    expect(response.body).not.toHaveProperty('backend_error');
  });
});
