import { createServer } from 'node:http';
import { createHmac } from 'node:crypto';
import type { Page, Route } from '@playwright/test';

type AdminUserRecord = {
  id: number;
  uuid: string;
  username: string;
  email: string;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  storage_quota_mb: number;
  storage_used_mb: number;
  created_at: string;
  updated_at: string;
  last_login?: string;
  metadata?: Record<string, unknown>;
};

const origin = (process.env.TLDW_ADMIN_UI_URL || 'http://127.0.0.1:3001').replace('localhost', '127.0.0.1');
const apiOrigin = new URL(process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5999');
const jwtSecret = process.env.JWT_SECRET_KEY || 'playwright-test-secret-1234567890';
export const singleUserApiKey = 'single-user-admin-key';

export const adminUser: AdminUserRecord = {
  id: 1,
  uuid: 'admin-user-1',
  username: 'admin',
  email: 'admin@example.com',
  role: 'admin',
  is_active: true,
  is_verified: true,
  storage_quota_mb: 4096,
  storage_used_mb: 256,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-02T00:00:00Z',
  last_login: '2026-03-10T01:00:00Z',
};

export const targetUser: AdminUserRecord = {
  id: 42,
  uuid: 'managed-user-42',
  username: 'managed-user',
  email: 'managed@example.com',
  role: 'user',
  is_active: true,
  is_verified: true,
  storage_quota_mb: 1024,
  storage_used_mb: 64,
  created_at: '2026-01-03T00:00:00Z',
  updated_at: '2026-03-09T12:00:00Z',
  metadata: {
    force_password_change: true,
  },
};

const jsonHeaders = {
  'access-control-allow-origin': '*',
  'content-type': 'application/json',
};

const toJsonResponse = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  });
};

const toBase64Url = (value: string): string => Buffer.from(value).toString('base64url');

const createJwt = (payload: Record<string, unknown>): string => {
  const header = toBase64Url(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const body = toBase64Url(JSON.stringify(payload));
  const signature = createHmac('sha256', jwtSecret)
    .update(`${header}.${body}`)
    .digest('base64url');

  return `${header}.${body}.${signature}`;
};

export const setAuthenticatedSession = async (page: Page) => {
  const now = Math.floor(Date.now() / 1000);
  const accessToken = createJwt({
    sub: String(adminUser.id),
    role: adminUser.role,
    iat: now,
    exp: now + 3600,
  });

  await page.context().addCookies([
    {
      name: 'access_token',
      value: accessToken,
      url: origin,
      httpOnly: true,
      sameSite: 'Lax',
    },
    {
      name: 'admin_session',
      value: '1',
      url: origin,
      sameSite: 'Lax',
    },
    {
      name: 'admin_auth_mode',
      value: 'jwt',
      url: origin,
      sameSite: 'Lax',
    },
  ]);
};

export const startSingleUserBackendStub = async (): Promise<() => Promise<void>> => {
  const server = createServer((request, response) => {
    if (request.url === '/api/v1/users/me' && request.headers['x-api-key'] === singleUserApiKey) {
      response.writeHead(200, jsonHeaders);
      response.end(JSON.stringify(adminUser));
      return;
    }

    response.writeHead(401, jsonHeaders);
    response.end(JSON.stringify({ detail: 'Invalid API key.' }));
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(Number(apiOrigin.port || '80'), apiOrigin.hostname, () => {
      server.off('error', reject);
      resolve();
    });
  });

  return async () =>
    await new Promise<void>((resolve, reject) => {
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      });
    });
};

export const installAdminApiRoutes = async (page: Page) => {
  await page.route('**/api/proxy/**', async (route) => {
    const url = new URL(route.request().url());
    const path = `${url.pathname}${url.search}`;

    if (path === '/api/proxy/users/me') {
      await toJsonResponse(route, adminUser);
      return;
    }

    if (path === '/api/proxy/admin/orgs') {
      await toJsonResponse(route, [{ id: 100, name: 'Acme Org', slug: 'acme-org' }]);
      return;
    }

    if (path === '/api/proxy/admin/users/1/effective-permissions') {
      await toJsonResponse(route, {
        permissions: ['read:users', 'write:users', 'read:orgs', 'write:orgs'],
      });
      return;
    }

    if (path === '/api/proxy/admin/users/42') {
      await toJsonResponse(route, targetUser);
      return;
    }

    if (path === '/api/proxy/admin/users/1/org-memberships') {
      await toJsonResponse(route, [{ org_id: 100, role: 'owner', org_name: 'Acme Org' }]);
      return;
    }

    if (path === '/api/proxy/admin/users/42/org-memberships') {
      await toJsonResponse(route, [{ org_id: 100, role: 'member', org_name: 'Acme Org' }]);
      return;
    }

    if (path === '/api/proxy/admin/users/42/mfa') {
      await toJsonResponse(route, {
        enabled: true,
        has_secret: true,
        has_backup_codes: true,
        method: 'totp',
      });
      return;
    }

    if (path === '/api/proxy/admin/users/42/sessions') {
      await toJsonResponse(route, []);
      return;
    }

    if (path.startsWith('/api/proxy/admin/audit-log?')) {
      await toJsonResponse(route, {
        entries: [
          {
            id: 'audit-1',
            timestamp: '2026-03-10T01:00:00Z',
            user_id: 42,
            action: 'login',
            resource: 'auth',
            ip_address: '127.0.0.1',
            details: {
              success: true,
              user_agent: 'Playwright',
            },
          },
        ],
        total: 1,
        limit: 20,
        offset: 0,
      });
      return;
    }

    if (path === '/api/proxy/admin/users/42/effective-permissions') {
      await toJsonResponse(route, {
        permissions: ['reports.read', 'admin.impersonate'],
      });
      return;
    }

    if (path === '/api/proxy/admin/users/42/overrides') {
      await toJsonResponse(route, { overrides: [] });
      return;
    }

    if (path === '/api/proxy/admin/permissions') {
      await toJsonResponse(route, []);
      return;
    }

    if (path === '/api/proxy/admin/users/42/rate-limits') {
      await toJsonResponse(route, {});
      return;
    }

    await toJsonResponse(route, { detail: `Unhandled smoke stub: ${path}` }, 404);
  });
};

export const installLoginRoutes = async (page: Page) => {
  await page.route('**/api/auth/login', async (route) => {
    const body = new URLSearchParams(route.request().postData() || '');
    if (body.get('username') !== 'admin' || body.get('password') !== 'AdminPass123!') {
      await toJsonResponse(route, { detail: 'Invalid username or password.' }, 401);
      return;
    }

    await toJsonResponse(route, {
      mfa_required: true,
      session_token: 'mfa-session-token',
      expires_in: 300,
      message: 'Enter your verification code to continue.',
    });
  });

  await page.route('**/api/auth/mfa/login', async (route) => {
    const raw = route.request().postData() || '{}';
    const body = JSON.parse(raw) as { session_token?: string; mfa_token?: string };
    if (body.session_token !== 'mfa-session-token' || body.mfa_token !== '123456') {
      await toJsonResponse(route, { detail: 'Invalid verification code.' }, 401);
      return;
    }

    await setAuthenticatedSession(page);
    await toJsonResponse(route, {
      token_type: 'bearer',
      expires_in: 3600,
    });
  });
};
