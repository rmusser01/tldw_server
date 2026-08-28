import { afterEach, describe, expect, it, vi } from 'vitest';

const originalRealBackendMode = process.env.TLDW_ADMIN_E2E_REAL_BACKEND;
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;
const originalJwtApiUrl = process.env.TLDW_ADMIN_E2E_JWT_API_URL;
const originalSingleUserApiUrl = process.env.TLDW_ADMIN_E2E_SINGLE_USER_API_URL;

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}

describe('buildApiUrlForRequest', () => {
  afterEach(() => {
    vi.resetModules();
    restoreEnv('TLDW_ADMIN_E2E_REAL_BACKEND', originalRealBackendMode);
    restoreEnv('NEXT_PUBLIC_API_URL', originalApiUrl);
    restoreEnv('TLDW_ADMIN_E2E_JWT_API_URL', originalJwtApiUrl);
    restoreEnv('TLDW_ADMIN_E2E_SINGLE_USER_API_URL', originalSingleUserApiUrl);
  });

  it('maps the single-user real-backend UI port to the single-user backend port in e2e mode', async () => {
    vi.resetModules();
    process.env.TLDW_ADMIN_E2E_REAL_BACKEND = 'true';
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8101';

    const { buildApiUrlForRequest } = await import('./api-config');

    expect(
      buildApiUrlForRequest(
        { url: 'http://127.0.0.1:3102/login' },
        '/users/me',
      ),
    ).toBe('http://127.0.0.1:8102/api/v1/users/me');
  });

  it('falls back to the configured API host when real-backend e2e mode is disabled', async () => {
    vi.resetModules();
    delete process.env.TLDW_ADMIN_E2E_REAL_BACKEND;
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8101';

    const { buildApiUrlForRequest } = await import('./api-config');

    expect(
      buildApiUrlForRequest(
        { url: 'http://127.0.0.1:3102/login' },
        '/users/me',
      ),
    ).toBe('http://127.0.0.1:8101/api/v1/users/me');
  });

  it('uses the configured project backend override in real-backend e2e mode', async () => {
    vi.resetModules();
    process.env.TLDW_ADMIN_E2E_REAL_BACKEND = 'true';
    process.env.TLDW_ADMIN_E2E_JWT_API_URL = 'http://127.0.0.1:9101';

    const { buildApiUrlForRequest } = await import('./api-config');

    expect(
      buildApiUrlForRequest(
        { url: 'http://127.0.0.1:3101/login' },
        '/users/me',
      ),
    ).toBe('http://127.0.0.1:9101/api/v1/users/me');
  });

  it('does not copy an untrusted request hostname into the backend URL', async () => {
    vi.resetModules();
    process.env.TLDW_ADMIN_E2E_REAL_BACKEND = 'true';
    process.env.TLDW_ADMIN_E2E_SINGLE_USER_API_URL = 'https://trusted-backend.example';

    const { buildApiUrlForRequest } = await import('./api-config');

    expect(
      buildApiUrlForRequest(
        { url: 'http://attacker.example:3102/login' },
        '/users/me',
      ),
    ).toBe('https://trusted-backend.example/api/v1/users/me');
  });
});
