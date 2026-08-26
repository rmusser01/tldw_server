import type { NextRequest } from 'next/server';
import { describe, it, expect, vi, beforeEach } from 'vitest';

const { mockBuildApiUrlForRequest } = vi.hoisted(() => ({
  mockBuildApiUrlForRequest: vi.fn(
    (_request: unknown, path: string) => `http://localhost:8000/api/v1${path}`,
  ),
}));

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
  buildApiUrlForRequest: mockBuildApiUrlForRequest,
}));

const request = { url: 'http://127.0.0.1:3102/api/health/ready' } as NextRequest;

describe('GET /api/health/ready', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    mockBuildApiUrlForRequest.mockClear();
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

  it('selects the backend health URL from the incoming UI request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const { GET } = await import('../route');

    await GET(request);

    expect(mockBuildApiUrlForRequest).toHaveBeenCalledWith(request, '/health');
  });
});
