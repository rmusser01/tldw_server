import { NextRequest, NextResponse } from 'next/server';
import { generateRequestId } from '@/lib/correlation-id';

export const ACCESS_TOKEN_COOKIE = 'access_token';
export const REFRESH_TOKEN_COOKIE = 'refresh_token';
export const API_KEY_COOKIE = 'x_api_key';
export const LEGACY_API_KEY_COOKIE = 'x-api-key';
export const SESSION_MARKER_COOKIE = 'admin_session';
export const AUTH_MODE_COOKIE = 'admin_auth_mode';

const isSecureCookie = process.env.NODE_ENV === 'production';

const buildHttpOnlyCookieOptions = (maxAge?: number) => ({
  httpOnly: true,
  sameSite: 'lax' as const,
  secure: isSecureCookie,
  path: '/',
  ...(typeof maxAge === 'number' ? { maxAge } : {}),
});

const buildMarkerCookieOptions = (maxAge?: number) => ({
  httpOnly: false,
  sameSite: 'lax' as const,
  secure: isSecureCookie,
  path: '/',
  ...(typeof maxAge === 'number' ? { maxAge } : {}),
});

const parseCookieValue = (value: string | undefined): string | null => {
  if (!value) return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
};

export const clearAdminSessionCookies = (response: NextResponse): void => {
  const expired = { maxAge: 0 };
  response.cookies.set({
    name: ACCESS_TOKEN_COOKIE,
    value: '',
    ...buildHttpOnlyCookieOptions(),
    ...expired,
  });
  response.cookies.set({
    name: REFRESH_TOKEN_COOKIE,
    value: '',
    ...buildHttpOnlyCookieOptions(),
    ...expired,
  });
  response.cookies.set({
    name: API_KEY_COOKIE,
    value: '',
    ...buildHttpOnlyCookieOptions(),
    ...expired,
  });
  response.cookies.set({
    name: LEGACY_API_KEY_COOKIE,
    value: '',
    ...buildHttpOnlyCookieOptions(),
    ...expired,
  });
  response.cookies.set({
    name: SESSION_MARKER_COOKIE,
    value: '',
    ...buildMarkerCookieOptions(),
    ...expired,
  });
  response.cookies.set({
    name: AUTH_MODE_COOKIE,
    value: '',
    ...buildMarkerCookieOptions(),
    ...expired,
  });
};

export const setJwtSessionCookies = (
  response: NextResponse,
  payload: {
    accessToken: string;
    refreshToken?: string;
    expiresIn?: number;
  }
): void => {
  clearAdminSessionCookies(response);
  response.cookies.set({
    name: ACCESS_TOKEN_COOKIE,
    value: payload.accessToken,
    ...buildHttpOnlyCookieOptions(payload.expiresIn),
  });
  if (payload.refreshToken) {
    response.cookies.set({
      name: REFRESH_TOKEN_COOKIE,
      value: payload.refreshToken,
      ...buildHttpOnlyCookieOptions(),
    });
  }
  response.cookies.set({
    name: SESSION_MARKER_COOKIE,
    value: '1',
    ...buildMarkerCookieOptions(payload.expiresIn),
  });
  response.cookies.set({
    name: AUTH_MODE_COOKIE,
    value: 'jwt',
    ...buildMarkerCookieOptions(payload.expiresIn),
  });
};

export const setApiKeySessionCookies = (
  response: NextResponse,
  apiKey: string
): void => {
  clearAdminSessionCookies(response);
  response.cookies.set({
    name: API_KEY_COOKIE,
    value: apiKey,
    ...buildHttpOnlyCookieOptions(),
  });
  response.cookies.set({
    name: SESSION_MARKER_COOKIE,
    value: '1',
    ...buildMarkerCookieOptions(),
  });
  response.cookies.set({
    name: AUTH_MODE_COOKIE,
    value: 'single_user',
    ...buildMarkerCookieOptions(),
  });
};

export const appendProxyHeaders = (request: NextRequest, headers: Headers): void => {
  const passthroughHeaders = [
    'accept',
    'content-type',
    'if-none-match',
    'if-modified-since',
    'range',
  ];

  for (const name of passthroughHeaders) {
    const value = request.headers.get(name);
    if (value) {
      headers.set(name, value);
    }
  }

  // Ensure every backend call carries a correlation ID —
  // middleware generates one for page routes, but /api routes
  // are excluded from middleware so we generate here when missing.
  headers.set('x-request-id', request.headers.get('x-request-id') || generateRequestId());
};

export const getBackendAuthHeaders = (request: NextRequest): Headers => {
  const headers = new Headers();
  const accessToken = parseCookieValue(request.cookies.get(ACCESS_TOKEN_COOKIE)?.value);
  const apiKeyCookie =
    parseCookieValue(request.cookies.get(API_KEY_COOKIE)?.value)
    ?? parseCookieValue(request.cookies.get(LEGACY_API_KEY_COOKIE)?.value);
  const apiKeyHeader = parseCookieValue(request.headers.get('x-api-key') ?? undefined);

  if (accessToken) {
    headers.set('Authorization', `Bearer ${accessToken}`);
    return headers;
  }

  if (apiKeyCookie) {
    headers.set('X-API-KEY', apiKeyCookie);
    return headers;
  }

  if (apiKeyHeader) {
    headers.set('X-API-KEY', apiKeyHeader);
  }

  return headers;
};

export const getRequestBody = async (request: NextRequest): Promise<BodyInit | undefined> => {
  if (request.method === 'GET' || request.method === 'HEAD') {
    return undefined;
  }

  const body = await request.arrayBuffer();
  return body.byteLength > 0 ? body : undefined;
};

export const buildProxyResponse = async (response: Response): Promise<NextResponse> => {
  const headers = new Headers();

  response.headers.forEach((value, key) => {
    const lowerKey = key.toLowerCase();
    if (['content-length', 'content-encoding', 'transfer-encoding', 'set-cookie'].includes(lowerKey)) {
      return;
    }
    headers.set(key, value);
  });

  return new NextResponse(response.body, {
    status: response.status,
    headers,
  });
};
