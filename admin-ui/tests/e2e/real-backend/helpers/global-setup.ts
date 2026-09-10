import { postAdminE2EJson } from './admin-e2e-support';
import {
  getProjectEnv,
  getRequestedRealBackendProjects,
  shouldManageBackend,
} from './project-env';
import { startManagedBackend, stopManagedBackend, waitForManagedBackend } from './backend-lifecycle';

type JwtSeedResponse = {
  users: {
    admin?: {
      key: string;
    };
  };
};

type JwtBootstrapResponse = {
  cookies: Array<{
    name: string;
    value: string;
  }>;
};

type ProxyCurrentUser = {
  id: number;
};

const wait = async (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

const warmUiRoute = async (
  baseUrl: string,
  path: string,
  init: RequestInit,
  acceptableStatuses: number[],
  timeoutMs = 60_000,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}${path}`, {
        ...init,
        headers: {
          Accept: 'application/json, text/html;q=0.9, */*;q=0.8',
          ...(init.headers || {}),
        },
      });
      if (acceptableStatuses.includes(response.status)) {
        return;
      }
    } catch {
      // Keep polling until timeout.
    }
    await wait(250);
  }

  throw new Error(`UI route ${path} did not become ready for ${baseUrl} within ${timeoutMs}ms`);
};

const warmJwtProtectedRoute = async (
  uiBaseUrl: string,
  apiBaseUrl: string,
  path: string,
): Promise<void> => {
  const seed = await postAdminE2EJson<JwtSeedResponse>(
    apiBaseUrl,
    '/api/v1/test-support/admin-e2e/seed',
    { scenario: 'jwt_admin' },
  );
  const principalKey = seed.users.admin?.key;
  if (!principalKey) {
    throw new Error('Missing seeded admin principal for JWT warm-up');
  }

  const bootstrap = await postAdminE2EJson<JwtBootstrapResponse>(
    apiBaseUrl,
    '/api/v1/test-support/admin-e2e/bootstrap-jwt-session',
    { principal_key: principalKey },
  );

  const cookieHeader = bootstrap.cookies
    .map((cookie) => `${cookie.name}=${cookie.value}`)
    .join('; ');

  await warmUiRoute(
    uiBaseUrl,
    '/api/proxy/users/me',
    {
      method: 'GET',
      headers: {
        Cookie: cookieHeader,
      },
    },
    [200],
  );

  let currentUserId: number | null = null;
  const currentUserDeadline = Date.now() + 15_000;
  while (Date.now() < currentUserDeadline) {
    try {
      const response = await fetch(`${uiBaseUrl}/api/proxy/users/me`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
          Cookie: cookieHeader,
        },
      });
      if (response.ok) {
        const body = await response.json() as ProxyCurrentUser;
        if (typeof body.id === 'number') {
          currentUserId = body.id;
          break;
        }
      }
    } catch {
      // Keep polling until timeout.
    }
    await wait(250);
  }

  if (currentUserId === null) {
    throw new Error(`Authenticated proxy route /api/proxy/users/me did not return a numeric user id for ${uiBaseUrl}`);
  }

  await warmUiRoute(
    uiBaseUrl,
    `/api/proxy/admin/users/${currentUserId}/effective-permissions`,
    {
      method: 'GET',
      headers: {
        Cookie: cookieHeader,
      },
    },
    [200],
  );

  await warmUiRoute(
    uiBaseUrl,
    '/api/proxy/admin/orgs',
    {
      method: 'GET',
      headers: {
        Cookie: cookieHeader,
      },
    },
    [200],
  );

  const deadline = Date.now() + 60_000;
  let lastStatus = 0;
  let lastUrl = '';
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${uiBaseUrl}${path}`, {
        method: 'GET',
        headers: {
          Accept: 'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8',
          Cookie: cookieHeader,
        },
        redirect: 'follow',
      });
      const body = await response.text();
      lastStatus = response.status;
      lastUrl = response.url;
      if (
        response.ok
        && !response.url.includes('/login')
        && !body.includes('Sign in to access the tldw_server Admin Panel')
      ) {
        return;
      }
    } catch {
      // Keep polling until timeout.
    }
    await wait(250);
  }

  throw new Error(
    `Protected UI route ${path} did not become ready for ${uiBaseUrl} within 60000ms (last status=${lastStatus}, last url=${lastUrl || 'unknown'})`,
  );
};

export default async function globalSetup(): Promise<void> {
  const requestedRealBackendProjects = getRequestedRealBackendProjects(process.argv);
  if (requestedRealBackendProjects.length === 0) {
    return;
  }

  for (const projectName of requestedRealBackendProjects) {
    const project = getProjectEnv(projectName);
    if (!shouldManageBackend(projectName)) {
      continue;
    }
    await stopManagedBackend(project).catch(() => undefined);
    await startManagedBackend(project);
  }

  for (const projectName of requestedRealBackendProjects) {
    const project = getProjectEnv(projectName);
    await waitForManagedBackend(project);
    await warmUiRoute(project.uiBaseUrl, '/login', { method: 'GET' }, [200]);
    await warmUiRoute(
      project.uiBaseUrl,
      '/api/proxy/users/me',
      { method: 'GET' },
      [200, 401, 403],
    );
    if (project.authMode === 'multi_user') {
      await warmUiRoute(
        project.uiBaseUrl,
        '/api/auth/login',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: 'username=probe&password=probe',
        },
        [200, 400, 401, 403],
      );
      await warmJwtProtectedRoute(project.uiBaseUrl, project.apiBaseUrl, '/data-ops');
    }
    if (project.authMode === 'single_user') {
      await warmUiRoute(
        project.uiBaseUrl,
        '/api/auth/apikey',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ apiKey: 'probe' }),
        },
        [200, 400, 401, 403],
      );
    }
  }
}
