import { NextRequest, NextResponse } from 'next/server';
import { buildApiUrlForRequest } from '@/lib/api-config';
import { logger } from '@/lib/logger';
import {
  clearAdminSessionCookies,
  getBackendAuthHeaders,
  ACCESS_TOKEN_COOKIE,
  API_KEY_COOKIE,
  LEGACY_API_KEY_COOKIE,
} from '@/lib/server-auth';
import { invalidateAuthCache } from '@/middleware';

export async function POST(request: NextRequest): Promise<NextResponse> {
  const headers = getBackendAuthHeaders(request);

  // Invalidate cached auth entries for the tokens being cleared.
  for (const name of [ACCESS_TOKEN_COOKIE, API_KEY_COOKIE, LEGACY_API_KEY_COOKIE]) {
    const value = request.cookies.get(name)?.value;
    if (value) {
      await invalidateAuthCache(value);
    }
  }

  try {
    await fetch(buildApiUrlForRequest(request, '/auth/logout'), {
      method: 'POST',
      headers,
      cache: 'no-store',
    });
  } catch (error) {
    logger.warn('Backend logout failed', {
      component: 'auth/logout',
      error: error instanceof Error ? error.message : String(error),
    });
  }

  const response = NextResponse.json({ ok: true });
  clearAdminSessionCookies(response);
  return response;
}
