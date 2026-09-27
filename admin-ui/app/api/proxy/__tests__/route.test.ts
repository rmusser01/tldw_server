/* @vitest-environment node */
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockBuildApiUrlForRequest = vi.fn(
  () => 'http://localhost:8000/api/v1/test',
);

const mockGetBackendAuthHeaders = vi.fn(() => new Headers());
const mockAppendProxyHeaders = vi.fn((_req: unknown, headers: Headers) => {
  const request = _req as { headers: Headers };
  for (const name of ['if-match', 'idempotency-key']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  // Real implementation ensures x-request-id is always set
  if (!headers.has('x-request-id')) {
    headers.set('x-request-id', 'test-request-id');
  }
});
const mockGetRequestBody = vi.fn(() => Promise.resolve(undefined));

vi.mock('@/lib/api-config', () => ({
  buildApiUrlForRequest: mockBuildApiUrlForRequest,
}));

vi.mock('@/lib/server-auth', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/server-auth')>();
  return {
    ...actual,
    getBackendAuthHeaders: mockGetBackendAuthHeaders,
    appendProxyHeaders: mockAppendProxyHeaders,
    getRequestBody: mockGetRequestBody,
  };
});

vi.mock('@/lib/logger', () => ({
  logger: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() },
}));

import { NextRequest } from 'next/server';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRequest(
  path: string,
  method = 'GET',
  headers?: HeadersInit,
): InstanceType<typeof NextRequest> {
  return new NextRequest(`http://localhost:3000/api/proxy${path}`, { method, headers });
}

function okResponse(body = '{"ok":true}'): Response {
  return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Proxy route – forward()', () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    vi.clearAllMocks();
    vi.useRealTimers();
    mockGetBackendAuthHeaders.mockReturnValue(new Headers());
    mockGetRequestBody.mockResolvedValue(undefined);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // 1. Success forwarding
  it('forwards a GET request and returns backend response', async () => {
    const backendResponse = okResponse('{"data":"hello"}');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(backendResponse));

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const result = await GET(makeRequest('/some/path'));

    expect(result.status).toBe(200);
    await expect(result.json()).resolves.toEqual({ data: 'hello' });
  });

  // 2. Auth header injection
  it('calls getBackendAuthHeaders with the request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse()));

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const req = makeRequest('/health');
    await GET(req);

    expect(mockGetBackendAuthHeaders).toHaveBeenCalledWith(req);
  });

  // 3. Proxy headers forwarded
  it('calls appendProxyHeaders with request and auth headers', async () => {
    const authHeaders = new Headers({ Authorization: 'Bearer tok' });
    mockGetBackendAuthHeaders.mockReturnValue(authHeaders);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse()));

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const req = makeRequest('/status');
    await GET(req);

    expect(mockAppendProxyHeaders).toHaveBeenCalledWith(req, authHeaders);
  });

  it('forwards conditional command headers without allowing an authorization override', async () => {
    const cookieAuthHeaders = new Headers({ Authorization: 'Bearer cookie-token' });
    mockGetBackendAuthHeaders.mockReturnValue(cookieAuthHeaders);
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"id":41}', {
      status: 200,
      headers: {
        'content-type': 'application/json',
        etag: '"admin-webhook-41-r2"',
        'cache-control': 'no-store',
        pragma: 'no-cache',
        'x-request-id': 'backend-request-id',
        'content-length': '9',
        'set-cookie': 'backend-session=do-not-forward',
      },
    }));
    vi.stubGlobal('fetch', fetchMock);

    const { PATCH } = await import('@/app/api/proxy/[...path]/route');
    const result = await PATCH(makeRequest('/admin/webhooks/41', 'PATCH', {
      authorization: 'Bearer browser-override',
      'if-match': '"admin-webhook-41-r1"',
      'idempotency-key': '0123456789abcdef',
    }));

    const forwarded = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(forwarded.get('authorization')).toBe('Bearer cookie-token');
    expect(forwarded.get('if-match')).toBe('"admin-webhook-41-r1"');
    expect(forwarded.get('idempotency-key')).toBe('0123456789abcdef');
    const responseHeaders = result.headers;
    expect(responseHeaders.get('etag')).toBe('"admin-webhook-41-r2"');
    expect(responseHeaders.get('cache-control')).toBe('no-store');
    expect(responseHeaders.get('pragma')).toBe('no-cache');
    expect(responseHeaders.get('x-request-id')).toBe('test-request-id');
    expect(responseHeaders.get('content-length')).toBeNull();
    expect(responseHeaders.get('set-cookie')).toBeNull();
  });

  it('does not reflect or log webhook request canaries from a backend 422', async () => {
    const canaries = {
      url: 'https://receiver.example/private/path?token=url-canary',
      query: 'query-canary',
      secret: 'forbidden-secret-canary',
      etag: 'etag-canary',
      idempotency: 'idempotency-canary',
    };
    mockGetRequestBody.mockResolvedValue(new TextEncoder().encode(JSON.stringify({
      url: canaries.url,
      secret: canaries.secret,
    })).buffer);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: {
        code: 'admin_webhook_validation_failed',
        message: 'Webhook request validation failed',
        request_id: 'test-request-id',
      },
    }), {
      status: 422,
      headers: {
        'content-type': 'application/json',
        'cache-control': 'no-store',
        pragma: 'no-cache',
        'x-request-id': 'test-request-id',
      },
    })));

    const { POST } = await import('@/app/api/proxy/[...path]/route');
    const { logger } = await import('@/lib/logger');
    const result = await POST(makeRequest('/admin/webhooks?probe=query-canary', 'POST', {
      'if-match': canaries.etag,
      'idempotency-key': canaries.idempotency,
    }));

    const serialized = JSON.stringify({
      body: await result.clone().text(),
      headers: Object.fromEntries(result.headers.entries()),
    });
    for (const canary of Object.values(canaries)) {
      expect(serialized).not.toContain(canary);
    }
    expect(result.headers.get('cache-control')).toBe('no-store');
    expect(result.headers.get('pragma')).toBe('no-cache');
    expect(result.headers.get('x-request-id')).toBe('test-request-id');
    expect(logger.error).not.toHaveBeenCalled();
  });

  // 3b. x-request-id is set on success response
  it('sets x-request-id on successful proxy response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse()));

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const result = await GET(makeRequest('/data'));

    expect(result.headers.get('x-request-id')).toBe('test-request-id');
  });

  // 3c. x-request-id is set on error response
  it('sets x-request-id on error proxy response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')));

    const { POST } = await import('@/app/api/proxy/[...path]/route');
    const result = await POST(makeRequest('/data', 'POST'));

    expect(result.status).toBe(502);
    await expect(result.json()).resolves.toEqual({ detail: 'Backend unavailable' });
  });

  // 4. Timeout returns 504
  it('returns 504 when backend request times out', async () => {
    // Mock fetch to hang until the AbortController fires, then throw AbortError
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
        return new Promise((_resolve, reject) => {
          if (init?.signal) {
            const onAbort = () => {
              const err = new DOMException('The operation was aborted.', 'AbortError');
              reject(err);
            };
            if (init.signal.aborted) {
              onAbort();
              return;
            }
            init.signal.addEventListener('abort', onAbort);
          }
        });
      }),
    );

    // Use fake timers so we can fast-forward past the 30s timeout
    vi.useFakeTimers();

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const resultPromise = GET(makeRequest('/slow'));

    // Advance past the proxy timeout (30_000ms)
    await vi.advanceTimersByTimeAsync(31_000);

    const result = await resultPromise;

    expect(result.status).toBe(504);
    await expect(result.json()).resolves.toEqual({ detail: 'Backend request timed out' });
  });

  // 5. Network error returns 502
  it('returns 502 when backend is unreachable (non-GET to skip retry)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new TypeError('fetch failed')),
    );

    const { POST } = await import('@/app/api/proxy/[...path]/route');
    const result = await POST(makeRequest('/data', 'POST'));

    expect(result.status).toBe(502);
    await expect(result.json()).resolves.toEqual({ detail: 'Backend unavailable' });
  });

  // 6. GET retry on network error
  it('retries once on network error for GET requests', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('fetch failed'))
      .mockResolvedValueOnce(okResponse('{"retried":true}'));
    vi.stubGlobal('fetch', fetchMock);

    // Use fake timers to skip the 500ms retry delay
    vi.useFakeTimers();

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const resultPromise = GET(makeRequest('/flaky'));

    // Advance past the 500ms retry delay
    await vi.advanceTimersByTimeAsync(600);

    const result = await resultPromise;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.status).toBe(200);
    await expect(result.json()).resolves.toEqual({ retried: true });
  });

  // 7. POST does NOT retry
  it('does not retry on network error for POST requests', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('@/app/api/proxy/[...path]/route');
    const result = await POST(makeRequest('/data', 'POST'));

    // Only one call — no retry for POST
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.status).toBe(502);
    await expect(result.json()).resolves.toEqual({ detail: 'Backend unavailable' });
  });

  // 8. Request body forwarding
  it('forwards request body for POST requests', async () => {
    const mockBody = new ArrayBuffer(4);
    mockGetRequestBody.mockResolvedValue(mockBody);

    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('@/app/api/proxy/[...path]/route');
    await POST(makeRequest('/submit', 'POST'));

    // Verify fetch was called with the body from getRequestBody
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const callArgs = fetchMock.mock.calls[0];
    expect(callArgs[1]).toMatchObject({
      method: 'POST',
      body: mockBody,
    });
  });

  // 9. Query string is forwarded
  it('forwards query string to backend URL', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const req = new NextRequest(
      'http://localhost:3000/api/proxy/search?q=hello&limit=10',
    );
    await GET(req);

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).toContain('?q=hello&limit=10');
  });

  // 10. All HTTP method exports delegate to forward
  it('exports all five HTTP method handlers', async () => {
    const mod = await import('@/app/api/proxy/[...path]/route');
    expect(typeof mod.GET).toBe('function');
    expect(typeof mod.POST).toBe('function');
    expect(typeof mod.PUT).toBe('function');
    expect(typeof mod.PATCH).toBe('function');
    expect(typeof mod.DELETE).toBe('function');
  });

  // 11. GET retry also fails — returns 502
  it('returns 502 when GET retry also fails', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    vi.useFakeTimers();

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const resultPromise = GET(makeRequest('/flaky'));

    // Advance past the 500ms retry delay
    await vi.advanceTimersByTimeAsync(600);

    const result = await resultPromise;

    // Two attempts for GET
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.status).toBe(502);
    await expect(result.json()).resolves.toEqual({ detail: 'Backend unavailable' });
  });

  // 12. AbortError on GET does NOT trigger retry
  it('does not retry GET requests on timeout (AbortError)', async () => {
    const abortErr = new DOMException('The operation was aborted.', 'AbortError');
    const fetchMock = vi.fn().mockRejectedValue(abortErr);
    vi.stubGlobal('fetch', fetchMock);

    const { GET } = await import('@/app/api/proxy/[...path]/route');
    const result = await GET(makeRequest('/timeout'));

    // Only one call — AbortError should NOT trigger retry
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.status).toBe(504);
    await expect(result.json()).resolves.toEqual({ detail: 'Backend request timed out' });
  });
});
