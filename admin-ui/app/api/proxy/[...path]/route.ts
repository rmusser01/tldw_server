import { NextRequest, NextResponse } from 'next/server';
import { buildApiUrlForRequest } from '@/lib/api-config';
import {
  appendProxyHeaders,
  buildProxyResponse,
  getBackendAuthHeaders,
  getRequestBody,
} from '@/lib/server-auth';
import { logger } from '@/lib/logger';

const PROXY_TIMEOUT_MS = 30_000;

const forward = async (request: NextRequest): Promise<NextResponse> => {
  const backendPath = request.nextUrl.pathname.replace(/^\/api\/proxy/, '') || '/';
  const backendUrl = new URL(buildApiUrlForRequest(request, backendPath));
  backendUrl.search = request.nextUrl.search;

  const headers = getBackendAuthHeaders(request);
  appendProxyHeaders(request, headers);
  const requestId = headers.get('x-request-id')
    ?? `proxy-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  headers.set('x-request-id', requestId);
  const body = await getRequestBody(request);
  const isGet = request.method === 'GET';

  const attemptFetch = async (): Promise<Response> => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), PROXY_TIMEOUT_MS);
    try {
      return await fetch(backendUrl.toString(), {
        method: request.method,
        headers,
        body,
        cache: 'no-store',
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }
  };

  try {
    const response = await attemptFetch();
    const proxyResponse = await buildProxyResponse(response);
    proxyResponse.headers.set('x-request-id', requestId);
    return proxyResponse;
  } catch (error) {
    let finalError: unknown = error;

    // Retry once for GET requests on network errors (not timeouts)
    if (isGet && error instanceof Error && error.name !== 'AbortError') {
      try {
        await new Promise((r) => setTimeout(r, 500));
        const response = await attemptFetch();
        const proxyResponse = await buildProxyResponse(response);
        proxyResponse.headers.set('x-request-id', requestId);
        return proxyResponse;
      } catch (retryError) {
        finalError = retryError;
      }
    }

    const isTimeout = finalError instanceof Error && finalError.name === 'AbortError';
    logger.error('Proxy request failed', {
      component: 'proxy',
      path: backendPath,
      method: request.method,
      requestId,
      error: finalError instanceof Error ? finalError.message : String(finalError),
      timeout: isTimeout,
    });

    return NextResponse.json(
      { detail: isTimeout ? 'Backend request timed out' : 'Backend unavailable' },
      { status: isTimeout ? 504 : 502, headers: { 'x-request-id': requestId } },
    );
  }
};

export async function GET(request: NextRequest): Promise<NextResponse> {
  return forward(request);
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  return forward(request);
}

export async function PUT(request: NextRequest): Promise<NextResponse> {
  return forward(request);
}

export async function PATCH(request: NextRequest): Promise<NextResponse> {
  return forward(request);
}

export async function DELETE(request: NextRequest): Promise<NextResponse> {
  return forward(request);
}
