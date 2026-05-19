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

describe('GET /api/health', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('returns ok status with timestamp and version', async () => {
    const { GET } = await import('../route');
    const response = await GET();
    expect(response.body).toMatchObject({ status: 'ok' });
    expect(response.body).toHaveProperty('timestamp');
    expect(response.body).toHaveProperty('version');
    expect(response.status).toBe(200);
  });

  it('returns Cache-Control no-store header', async () => {
    const { GET } = await import('../route');
    const response = await GET();
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });
});
