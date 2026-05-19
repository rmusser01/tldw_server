/**
 * Next.js instrumentation hook — runs once when the server starts.
 * @see https://nextjs.org/docs/app/building-your-application/optimizing/instrumentation
 */
export async function register() {
  // Validate environment variables at startup so misconfiguration fails fast
  // rather than silently at request time.
  if (process.env.NODE_ENV === 'production') {
    // In production Docker, NEXT_PUBLIC_* vars are build-time only (baked into
    // the client bundle) and not available at runtime.  Validate server-side
    // secrets instead.
    const { validateRuntimeEnv } = await import('@/lib/env');
    validateRuntimeEnv();
  } else {
    // In development, NEXT_PUBLIC_* vars are available via .env files.
    const { validateEnv } = await import('@/lib/env');
    validateEnv();
  }
}
