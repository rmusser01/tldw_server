export const ADMIN_E2E_SUPPORT_HEADER = 'X-TLDW-Admin-E2E-Key';
export const ADMIN_E2E_SUPPORT_KEY_ENV = 'TLDW_ADMIN_E2E_SUPPORT_KEY';
export const DEFAULT_ADMIN_E2E_SUPPORT_KEY = 'playwright-admin-e2e-support-key';

export const getAdminE2ESupportKey = (): string =>
  process.env[ADMIN_E2E_SUPPORT_KEY_ENV]?.trim() || DEFAULT_ADMIN_E2E_SUPPORT_KEY;

export const getAdminE2ESupportHeaders = (): Record<string, string> => ({
  [ADMIN_E2E_SUPPORT_HEADER]: getAdminE2ESupportKey(),
});

export const postAdminE2EJson = async <T>(
  baseUrl: string,
  path: string,
  body?: unknown,
): Promise<T> => {
  const bootstrap = await fetch(`${baseUrl}/health`);
  if (!bootstrap.ok) {
    const detail = await bootstrap.text().catch(() => '');
    throw new Error(`Admin e2e CSRF bootstrap failed: ${bootstrap.status} ${detail}`.trim());
  }
  // Node fetch has no cookie jar; the middleware returns the cookie token in this header too.
  const csrfToken = bootstrap.headers.get('X-CSRF-Token');
  const response = await fetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers: {
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      ...getAdminE2ESupportHeaders(),
      ...(csrfToken ? {
        Cookie: `csrf_token=${csrfToken}`,
        'X-CSRF-Token': csrfToken,
      } : {}),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`Admin e2e request failed for ${path}: ${response.status} ${detail}`.trim());
  }

  return response.json() as Promise<T>;
};
