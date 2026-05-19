import { NextRequest, NextResponse } from 'next/server';
import { buildApiUrlForRequest } from '@/lib/api-config';
import { setJwtSessionCookies } from '@/lib/server-auth';
import { checkRateLimit, extractClientIp } from '@/lib/rate-limiter';

type LoginResponsePayload = {
  access_token?: string;
  refresh_token?: string;
  token_type?: string;
  expires_in?: number;
  session_token?: string;
  mfa_required?: boolean;
  message?: string;
};

const sanitizePayload = (payload: LoginResponsePayload): Omit<LoginResponsePayload, 'access_token' | 'refresh_token'> => {
  const sanitized = { ...payload };
  delete sanitized.access_token;
  delete sanitized.refresh_token;
  return sanitized;
};

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

  const body = await request.text();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10_000);

  let response: Response;
  try {
    response = await fetch(buildApiUrlForRequest(request, '/auth/login'), {
      method: 'POST',
      headers: {
        'Content-Type': request.headers.get('content-type') ?? 'application/x-www-form-urlencoded',
      },
      body,
      cache: 'no-store',
      signal: controller.signal,
    });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return NextResponse.json({ detail: 'Login request timed out' }, { status: 504 });
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  const payload = await response.json().catch(() => null) as LoginResponsePayload | null;
  if (!response.ok || !payload) {
    return NextResponse.json(payload ?? { detail: 'Login failed' }, { status: response.status });
  }

  const nextResponse = NextResponse.json(sanitizePayload(payload), { status: response.status });
  if (typeof payload.access_token === 'string') {
    setJwtSessionCookies(nextResponse, {
      accessToken: payload.access_token,
      refreshToken: payload.refresh_token,
      expiresIn: payload.expires_in,
    });
  }

  return nextResponse;
}
