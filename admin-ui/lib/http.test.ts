/* @vitest-environment jsdom */
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./auth', () => ({
  getApiKey: vi.fn(() => null),
  getJWTToken: vi.fn(() => null),
  logout: vi.fn(() => Promise.resolve()),
}));

describe('http auth transport', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ ok: true }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )));
  });

  it('routes JSON requests through the same-origin proxy without browser bearer headers', async () => {
    const { requestJson } = await import('./http');

    await requestJson('/users/me');

    expect(fetch).toHaveBeenCalledWith('/api/proxy/users/me', expect.objectContaining({
      credentials: 'include',
      headers: expect.any(Headers),
    }));

    const [, init] = vi.mocked(fetch).mock.calls[0] ?? [];
    const headers = new Headers(init?.headers);
    expect(headers.get('Authorization')).toBeNull();
    expect(headers.get('X-API-KEY')).toBeNull();
  });

  it('forwards in-memory API keys only to the same-origin proxy', async () => {
    const auth = await import('./auth');
    vi.mocked(auth.getApiKey).mockReturnValue('ephemeral-api-key');

    const { requestJson } = await import('./http');
    await requestJson('/users/me');

    const [, init] = vi.mocked(fetch).mock.calls[0] ?? [];
    const headers = new Headers(init?.headers);
    expect(headers.get('X-API-KEY')).toBe('ephemeral-api-key');
  });

  it('returns bounded response metadata without changing requestJson results', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => Promise.resolve(new Response(
      JSON.stringify({ id: 41, revision: 2 }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          ETag: '"admin-webhook-41-r2"',
          'Retry-After': '17',
          'X-Request-ID': 'request-41',
        },
      }
    ))));
    const { requestJson, requestJsonWithMetadata } = await import('./http');

    await expect(requestJson<{ id: number }>('/admin/webhooks/41')).resolves.toEqual({
      id: 41,
      revision: 2,
    });
    await expect(
      requestJsonWithMetadata<{ id: number; revision: number }>('/admin/webhooks/41')
    ).resolves.toEqual({
      data: { id: 41, revision: 2 },
      status: 200,
      etag: '"admin-webhook-41-r2"',
      requestId: 'request-41',
      retryAfterSeconds: 17,
    });
  });

  it('does not infer a JSON content type for text request bodies', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('ok', {
      status: 200,
      headers: { 'Content-Type': 'text/plain' },
    })));
    const { requestText } = await import('./http');

    await expect(requestText('/metrics/text', {
      method: 'POST',
      body: 'raw metrics command',
    })).resolves.toBe('ok');

    const [, init] = vi.mocked(fetch).mock.calls[0] ?? [];
    expect(new Headers(init?.headers).get('content-type')).toBeNull();
  });

  it.each([409, 412, 422, 428, 503])(
    'parses a canonical webhook %i error without retaining raw response detail',
    async (status) => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
        JSON.stringify({
          error: {
            code: 'admin_webhook_precondition_failed',
            message: 'Webhook precondition failed',
            request_id: 'request-42',
          },
        }),
        {
          status,
          headers: {
            'Content-Type': 'application/json',
            'X-Request-ID': 'request-42',
          },
        }
      )));
      const { requestJson, WebhookApiError } = await import('./http');

      const error = await requestJson('/admin/webhooks/41').catch((caught) => caught);

      expect(error).toBeInstanceOf(WebhookApiError);
      expect(error).toMatchObject({
        status,
        code: 'admin_webhook_precondition_failed',
        message: 'Webhook precondition failed',
        requestId: 'request-42',
        detail: undefined,
      });
      expect(error).not.toHaveProperty('responseText');
    }
  );

  it('preserves bounded canonical conflicts for the incident webhook command route', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: {
        code: 'admin_webhook_idempotency_conflict',
        message: 'Idempotency key conflicts with another command',
        request_id: 'request-incident-conflict',
      },
    }), {
      status: 409,
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': 'request-incident-conflict',
      },
    })));
    const { requestJson, WebhookApiError } = await import('./http');

    const error = await requestJson('/admin/incidents/inc-41/notify-webhooks', {
      method: 'POST',
    }).catch((caught) => caught);

    expect(error).toBeInstanceOf(WebhookApiError);
    expect(error).toMatchObject({
      status: 409,
      code: 'admin_webhook_idempotency_conflict',
      message: 'Idempotency key conflicts with another command',
      requestId: 'request-incident-conflict',
      detail: undefined,
    });
  });

  it('rejects malformed canonical errors without retaining a response canary', async () => {
    const canary = 'forbidden-response-canary';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: canary }),
      { status: 422, headers: { 'Content-Type': 'application/json' } }
    )));
    const { requestJson, WebhookContractError } = await import('./http');

    const error = await requestJson('/admin/webhooks').catch((caught) => caught);

    expect(error).toBeInstanceOf(WebhookContractError);
    expect(error).toMatchObject({ status: 422, detail: undefined });
    expect(error.message).not.toContain(canary);
    expect(JSON.stringify(error)).not.toContain(canary);
  });

  it('rejects a canonical error whose body and header request IDs disagree', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: {
        code: 'admin_webhook_precondition_failed',
        message: 'Webhook precondition failed',
        request_id: 'request-body',
      },
    }), {
      status: 412,
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': 'request-header',
      },
    })));
    const { requestJson, WebhookContractError } = await import('./http');

    await expect(requestJson('/admin/webhooks/41')).rejects.toBeInstanceOf(
      WebhookContractError
    );
  });

  it('redacts malformed canonical success JSON from parse failures', async () => {
    const canary = 'malformed-success-canary';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(`{"value":"${canary}`, {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        ETag: '"admin-webhook-41-r2"',
        'X-Request-ID': 'request-success',
      },
    })));
    const { requestJsonWithMetadata, WebhookContractError } = await import('./http');

    const error = await requestJsonWithMetadata('/admin/webhooks/41').catch(
      (caught) => caught
    );

    expect(error).toBeInstanceOf(WebhookContractError);
    expect(error.message).not.toContain(canary);
    expect(JSON.stringify(error)).not.toContain(canary);
  });

  it.each([
    [502, 'Backend unavailable', 'Webhook backend is unavailable'],
    [504, 'Backend request timed out', 'Webhook backend request timed out'],
  ])(
    'maps an exact proxy %i response to a retryable bounded transport error',
    async (status, detail, message) => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
        JSON.stringify({ detail }),
        {
          status,
          headers: {
            'Content-Type': 'application/json',
            'X-Request-ID': 'proxy-request-1',
          },
        },
      )));
      const { requestJson, WebhookTransportError } = await import('./http');

      const error = await requestJson('/admin/webhooks').catch((caught) => caught);

      expect(error).toBeInstanceOf(WebhookTransportError);
      expect(error).toMatchObject({
        status,
        message,
        requestId: 'proxy-request-1',
        detail: undefined,
      });
    },
  );

  it('preserves generic ApiError behavior for non-webhook endpoints', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: 'generic detail' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    )));
    const { ApiError, requestJson } = await import('./http');

    const error = await requestJson('/admin/users').catch((caught) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 400,
      message: 'generic detail',
      detail: { detail: 'generic detail' },
    });
  });
});
