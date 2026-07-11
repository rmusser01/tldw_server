import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import handler from '@web/pages/api/_tldw-webui/session';
import { createApiRequest, createApiResponse, type MockApiResponse } from './test-utils';

const API_KEY = 'runtime-single-user-key';
const WEB_ORIGIN = 'http://127.0.0.1:8080';
const BACKEND_ORIGIN = 'http://app:8000';
const SESSION_ENDPOINT = `${BACKEND_ORIGIN}/api/v1/auth/single-user/session`;

const ORIGINAL_ENV = {
  AUTH_MODE: process.env.AUTH_MODE,
  SINGLE_USER_API_KEY: process.env.SINGLE_USER_API_KEY,
  TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH,
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  TLDW_INTERNAL_API_ORIGIN: process.env.TLDW_INTERNAL_API_ORIGIN,
  SINGLE_USER_SESSION_COOKIE_NAME: process.env.SINGLE_USER_SESSION_COOKIE_NAME,
};

const restoreEnv = () => {
  for (const [key, value] of Object.entries(ORIGINAL_ENV)) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
};

const configureRuntimeAuth = () => {
  process.env.AUTH_MODE = 'single_user';
  process.env.SINGLE_USER_API_KEY = API_KEY;
  process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH = '1';
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart';
  process.env.TLDW_INTERNAL_API_ORIGIN = BACKEND_ORIGIN;
  delete process.env.SINGLE_USER_SESSION_COOKIE_NAME;
};

const backendResponse = ({
  status = 200,
  cookies = [],
  headers = {},
}: {
  status?: number;
  cookies?: string[];
  headers?: Record<string, string>;
} = {}) => {
  const responseHeaders = new Headers(headers);
  for (const cookie of cookies) responseHeaders.append('Set-Cookie', cookie);
  return {
    status,
    headers: responseHeaders,
  } as Response;
};

const makeRequest = ({
  method = 'POST',
  headers = {},
  remoteAddress = '127.0.0.1',
  encrypted = false,
}: {
  method?: string;
  headers?: Record<string, string>;
  remoteAddress?: string | null;
  encrypted?: boolean;
} = {}) => {
  const req = createApiRequest({
    method,
    url: '/api/_tldw-webui/session',
    headers: {
      host: '127.0.0.1:8080',
      origin: WEB_ORIGIN,
      'sec-fetch-site': 'same-origin',
      ...headers,
    },
  });
  if (remoteAddress !== null) {
    Object.defineProperty(req, 'socket', {
      configurable: true,
      value: { encrypted, remoteAddress },
    });
  }
  return req;
};

const callRoute = async (
  requestOptions: Parameters<typeof makeRequest>[0] = {}
): Promise<MockApiResponse> => {
  const res = createApiResponse();
  await handler(makeRequest(requestOptions), res);
  return res;
};

describe('WebUI runtime session bootstrap API', () => {
  const mockFetch = vi.fn<typeof fetch>();

  beforeEach(() => {
    restoreEnv();
    configureRuntimeAuth();
    mockFetch.mockReset();
    mockFetch.mockResolvedValue(backendResponse());
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    restoreEnv();
  });

  it('posts to the fixed internal endpoint with only filtered browser headers and the server key', async () => {
    const res = await callRoute({
      headers: {
        authorization: 'Bearer browser-secret',
        cookie: 'ignored=secret; tldw_single_user_session=existing==; csrf_token=csrf-value',
        'user-agent': 'runtime-session-test',
        'x-api-key': 'browser-key',
        'x-internal-api-origin': 'http://attacker.test',
      },
    });

    expect(res.statusCode).toBe(200);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(SESSION_ENDPOINT, {
      method: 'POST',
      headers: {
        Cookie: 'tldw_single_user_session=existing==; csrf_token=csrf-value',
        'User-Agent': 'runtime-session-test',
        'X-API-KEY': API_KEY,
      },
      redirect: 'manual',
      signal: expect.any(AbortSignal),
    });
  });

  it('forwards only the configured session cookie and csrf cookie inbound', async () => {
    process.env.SINGLE_USER_SESSION_COOKIE_NAME = 'custom_session';

    const res = await callRoute({
      headers: {
        cookie:
          'tldw_single_user_session=drop-default; custom_session=keep-custom; csrf_token=keep-csrf; unrelated=drop',
      },
    });

    expect(res.statusCode).toBe(200);
    expect(mockFetch).toHaveBeenCalledWith(
      SESSION_ENDPOINT,
      expect.objectContaining({
        headers: {
          Cookie: 'custom_session=keep-custom; csrf_token=keep-csrf',
          'X-API-KEY': API_KEY,
        },
      })
    );
  });

  it('forwards separate auth and csrf cookies while preserving safe attributes', async () => {
    const cookies = [
      'tldw_single_user_session=opaque; Path=/api; HttpOnly; SameSite=Lax; Max-Age=2592000',
      'csrf_token=csrf; Path=/; SameSite=Lax; Secure',
      'unrelated=drop-me; Path=/; HttpOnly',
      'tldw_single_user_session=rotated; Path=/api; HttpOnly; SameSite=Lax',
    ];
    mockFetch.mockResolvedValue(backendResponse({ cookies }));

    const res = await callRoute();

    expect(res.statusCode).toBe(200);
    expect(res.headers['set-cookie']).toEqual([cookies[0], cookies[1], cookies[3]]);
  });

  it('forwards only the configured session cookie and csrf cookie outbound', async () => {
    process.env.SINGLE_USER_SESSION_COOKIE_NAME = 'custom_session';
    const cookies = [
      'tldw_single_user_session=drop-default; Path=/api; HttpOnly; SameSite=Lax',
      'custom_session=keep-custom; Path=/api; HttpOnly; SameSite=Lax',
      'csrf_token=keep-csrf; Path=/; SameSite=Lax',
      'unrelated=drop; Path=/; HttpOnly',
    ];
    mockFetch.mockResolvedValue(backendResponse({ cookies }));

    const res = await callRoute();

    expect(res.statusCode).toBe(200);
    expect(res.headers['set-cookie']).toEqual([cookies[1], cookies[2]]);
  });

  it('copies only safe response metadata and never returns a secret body', async () => {
    mockFetch.mockResolvedValue(
      backendResponse({
        status: 201,
        cookies: ['tldw_single_user_session=opaque; Path=/api; HttpOnly; SameSite=Lax'],
        headers: {
          'cache-control': 'no-store',
          'content-type': 'application/json',
          'x-api-key': API_KEY,
          'x-backend-diagnostic': 'opaque',
        },
      })
    );

    const res = await callRoute();

    expect(res.statusCode).toBe(201);
    expect(res.headers).toMatchObject({
      'cache-control': 'no-store',
      'content-type': 'application/json',
    });
    expect(res.headers).not.toHaveProperty('x-api-key');
    expect(res.headers).not.toHaveProperty('x-backend-diagnostic');
    expect(res.body).toBeUndefined();
    expect(JSON.stringify(res.body ?? '')).not.toContain(API_KEY);
    expect(JSON.stringify(res.body ?? '')).not.toContain('opaque');
  });

  it.each(['GET', 'PUT', 'DELETE', 'PATCH'])(
    'rejects the %s method without contacting the backend',
    async (method) => {
      const res = await callRoute({ method });

      expect(res.statusCode).toBe(405);
      expect(res.headers.allow).toBe('POST');
      expect(res.body).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    }
  );

  it.each([
    ['missing', ''],
    ['null', 'null'],
    ['malformed', 'not-an-origin'],
    ['cross-origin host', 'http://attacker.test'],
    ['cross-origin scheme', 'https://127.0.0.1:8080'],
    ['cross-origin port', 'http://127.0.0.1:8081'],
  ])('rejects a %s Origin', async (_name, origin) => {
    const res = await callRoute({ headers: { origin } });

    expect(res.statusCode).toBe(403);
    expect(res.body).toBeUndefined();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('accepts an exact HTTPS origin when the direct request socket is encrypted', async () => {
    const res = await callRoute({
      encrypted: true,
      headers: { origin: 'https://127.0.0.1:8080' },
    });

    expect(res.statusCode).toBe(200);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it.each(['cross-site', 'same-site', 'none'])('rejects %s Fetch Metadata', async (fetchSite) => {
    const res = await callRoute({
      headers: { 'sec-fetch-site': fetchSite },
    });

    expect(res.statusCode).toBe(403);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('allows an omitted Fetch Metadata header when Origin is exact', async () => {
    const req = makeRequest();
    delete req.headers['sec-fetch-site'];
    const res = createApiResponse();

    await handler(req, res);

    expect(res.statusCode).toBe(200);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['wrong auth mode', { AUTH_MODE: 'multi_user' }],
    ['non-exact exposure flag', { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: 'true' }],
    ['wrong deployment mode', { NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: 'advanced' }],
    ['missing key', { SINGLE_USER_API_KEY: '' }],
    ['placeholder key', { SINGLE_USER_API_KEY: 'change-me' }],
  ])('fails closed for %s', async (_name, envPatch) => {
    Object.assign(process.env, envPatch);

    const res = await callRoute();

    expect(res.statusCode).toBe(503);
    expect(res.body).toBeUndefined();
    expect(mockFetch).not.toHaveBeenCalled();
    expect(JSON.stringify(res.body ?? '')).not.toContain(API_KEY);
  });

  it.each([
    '',
    'csrf_token',
    '__Host-session',
    '__Http-session',
    '__secure-session',
    'invalid name',
    'session=value',
    'session;name',
    '/session',
  ])(
    'fails closed without forwarding the key for invalid cookie name %j',
    async (cookieName) => {
      process.env.SINGLE_USER_SESSION_COOKIE_NAME = cookieName;

      const res = await callRoute();

      expect(res.statusCode).toBe(503);
      expect(res.body).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
      expect(JSON.stringify(res.body ?? '')).not.toContain(API_KEY);
    }
  );

  it.each([
    ['non-loopback host', { headers: { host: 'webui.example.test' } }],
    ['untrusted peer', { remoteAddress: '203.0.113.10' }],
    ['missing peer', { remoteAddress: null }],
    ['forwarding headers', { headers: { 'x-forwarded-for': '127.0.0.1' } }],
  ])('fails closed for %s', async (_name, requestOptions) => {
    const res = await callRoute(requestOptions);

    expect(res.statusCode).toBe(503);
    expect(res.body).toBeUndefined();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it.each([
    ['missing', ''],
    ['relative', '/api'],
    ['non-HTTP', 'ftp://app:8000'],
    ['credential-bearing', 'http://user:pass@app:8000'],
    ['path-bearing', 'http://app:8000/backend'],
    ['query-bearing', 'http://app:8000/?target=other'],
    ['fragment-bearing', 'http://app:8000/#backend'],
    ['empty-query marker', 'http://app:8000?'],
    ['empty-fragment marker', 'http://app:8000#'],
    ['dot-segment path', 'http://app:8000/./'],
    ['collapsed dot-segment path', 'http://app:8000/a/../'],
    ['noncanonical host case', 'http://APP:8000'],
    ['default port', 'http://app:80'],
    ['surrounding whitespace', ' http://app:8000 '],
  ])('fails closed for a %s internal origin', async (_name, origin) => {
    process.env.TLDW_INTERNAL_API_ORIGIN = origin;

    const res = await callRoute();

    expect(res.statusCode).toBe(503);
    expect(res.body).toBeUndefined();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('forwards backend authentication failures without their body or diagnostics', async () => {
    mockFetch.mockResolvedValue(
      backendResponse({
        status: 401,
        headers: {
          'content-type': 'application/json',
          'x-backend-diagnostic': API_KEY,
        },
      })
    );

    const res = await callRoute();

    expect(res.statusCode).toBe(401);
    expect(res.body).toBeUndefined();
    expect(JSON.stringify(res.headers)).not.toContain(API_KEY);
  });

  it('returns a generic gateway failure when the backend is unavailable', async () => {
    mockFetch.mockRejectedValue(new Error(`backend failed with ${API_KEY}`));

    const res = await callRoute();

    expect(res.statusCode).toBe(502);
    expect(res.body).toBeUndefined();
    expect(JSON.stringify(res.body ?? '')).not.toContain(API_KEY);
  });

  it('returns a generic timeout without exposing diagnostics', async () => {
    mockFetch.mockRejectedValue(new DOMException(`timed out with ${API_KEY}`, 'TimeoutError'));

    const res = await callRoute();

    expect(res.statusCode).toBe(504);
    expect(res.body).toBeUndefined();
    expect(JSON.stringify(res.body ?? '')).not.toContain(API_KEY);
  });

  it('fails closed when the backend response cannot provide separate Set-Cookie values', async () => {
    mockFetch.mockResolvedValue({
      status: 200,
      headers: new Map(),
    } as unknown as Response);

    const res = await callRoute();

    expect(res.statusCode).toBe(502);
    expect(res.body).toBeUndefined();
    expect(res.headers).not.toHaveProperty('set-cookie');
  });

  it('does not partially set cookies from a malformed backend response', async () => {
    mockFetch.mockResolvedValue({
      status: 200,
      headers: {
        getSetCookie: () => ['tldw_single_user_session=opaque; Path=/api; HttpOnly; SameSite=Lax'],
        get: () => {
          throw new TypeError('malformed');
        },
      },
    } as unknown as Response);

    const res = await callRoute();

    expect(res.statusCode).toBe(502);
    expect(res.body).toBeUndefined();
    expect(res.headers).not.toHaveProperty('set-cookie');
  });

  it('fails closed for a malformed backend status', async () => {
    mockFetch.mockResolvedValue({
      status: 0,
      headers: new Headers(),
    } as Response);

    const res = await callRoute();

    expect(res.statusCode).toBe(502);
    expect(res.body).toBeUndefined();
  });
});
