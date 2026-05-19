import { NextResponse } from 'next/server';
import { buildApiUrl } from '@/lib/api-config';

export async function GET(): Promise<NextResponse> {
  const timestamp = new Date().toISOString();

  // Probe backend health with 2-second timeout
  let backendReachable = false;
  let backendError: string | null = null;
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  try {
    const controller = new AbortController();
    timeoutId = setTimeout(() => controller.abort(), 2000);
    const response = await fetch(buildApiUrl('/health'), {
      method: 'GET',
      cache: 'no-store',
      signal: controller.signal,
    });
    backendReachable = response.ok;
    if (!response.ok) {
      backendError = `Backend returned ${response.status}`;
    }
  } catch (error) {
    // Log the full error server-side but don't expose internals to clients
    console.error('[health/ready] Backend probe failed:', error);
    backendError = 'Backend unreachable';
  } finally {
    if (timeoutId !== undefined) clearTimeout(timeoutId);
  }

  const status = backendReachable ? 'ready' : 'not_ready';
  const httpStatus = backendReachable ? 200 : 503;

  return NextResponse.json(
    {
      status,
      timestamp,
      backend: backendReachable ? 'reachable' : 'unreachable',
      ...(backendError ? { backend_error: backendError } : {}),
    },
    { status: httpStatus, headers: { 'Cache-Control': 'no-store' } }
  );
}
