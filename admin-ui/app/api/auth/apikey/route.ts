import { NextRequest, NextResponse } from 'next/server';
import { buildApiUrlForRequest } from '@/lib/api-config';
import { setApiKeySessionCookies } from '@/lib/server-auth';
import { checkRateLimit, extractClientIp } from '@/lib/rate-limiter';

const isAdminApiKeyLoginEnabled = (): boolean =>
  process.env.ADMIN_UI_ALLOW_API_KEY_LOGIN === 'true';

const isEnterpriseAdminUiMode = (): boolean =>
  process.env.ADMIN_UI_ENTERPRISE_MODE === 'true';

const isSingleUserAuthMode = (): boolean =>
  process.env.AUTH_MODE === 'single_user';

const shouldAttachTestDiagnostics = (): boolean =>
  process.env.TEST_MODE === 'true';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const clientIp = extractClientIp(request.headers);
  if (clientIp !== 'unknown') {
    const rateCheck = checkRateLimit(clientIp);
    if (!rateCheck.allowed) {
      return NextResponse.json(
        { detail: 'Too many login attempts. Please try again later.' },
        {
          status: 429,
          headers: { 'Retry-After': String(rateCheck.retryAfterSeconds) },
        }
      );
    }
  }

  if (isEnterpriseAdminUiMode() || !isAdminApiKeyLoginEnabled() || !isSingleUserAuthMode()) {
    return NextResponse.json(
      { detail: 'Admin UI API key login is disabled. Use multi-user credentials.' },
      { status: 403 }
    );
  }

  const body = await request.json().catch(() => null) as { apiKey?: string } | null;
  const apiKey = body?.apiKey?.trim();

  if (!apiKey) {
    return NextResponse.json({ detail: 'API key is required' }, { status: 400 });
  }

  const backendUrl = buildApiUrlForRequest(request, '/users/me');
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10_000);

  let response: Response;
  try {
    response = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        'X-API-KEY': apiKey,
      },
      cache: 'no-store',
      signal: controller.signal,
    });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return NextResponse.json({ detail: 'API key validation timed out' }, { status: 504 });
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload) {
    const nextResponse = NextResponse.json(
      payload ?? { detail: 'API key validation failed' },
      { status: response.status },
    );
    if (shouldAttachTestDiagnostics()) {
      nextResponse.headers.set('X-TLDW-Backend-Url', backendUrl);
    }
    return nextResponse;
  }

  const nextResponse = NextResponse.json({ user: payload }, { status: 200 });
  if (shouldAttachTestDiagnostics()) {
    nextResponse.headers.set('X-TLDW-Backend-Url', backendUrl);
  }
  setApiKeySessionCookies(nextResponse, apiKey);
  return nextResponse;
}
