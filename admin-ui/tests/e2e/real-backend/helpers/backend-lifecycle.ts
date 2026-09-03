import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, realpathSync } from 'node:fs';
import { resolve } from 'node:path';

import { DEFAULT_ADMIN_E2E_SUPPORT_KEY } from './admin-e2e-support';
import { getFixturePasswordEnv } from './fixture-secrets';
import { type RealBackendProjectEnv } from './project-env';

export type ManagedBackendProcess = {
  process: ChildProcess;
  command: string;
  args: string[];
};

const repoRoot = resolve(process.cwd(), '..');
const lifecycleScript = resolve(repoRoot, 'tldw_Server_API/scripts/server_lifecycle.py');
const tempRoot = realpathSync('/tmp');
const MANAGED_JWT_WEBHOOK_ENV = {
  TLDW_ADMIN_WEBHOOKS_MODE: 'on',
  TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV: 'true',
  TLDW_ADMIN_WEBHOOKS_E2E_LOOPBACK: 'true',
  TLDW_ADMIN_WEBHOOK_KEYS_JSON:
    '{"admin-e2e-primary":"d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3c="}',
  TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID: 'admin-e2e-primary',
  TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS: '1',
  TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_FRESHNESS_SECONDS: '10',
};

const getPythonCommand = (): string => {
  if (process.env.TLDW_ADMIN_E2E_PYTHON) {
    return process.env.TLDW_ADMIN_E2E_PYTHON;
  }

  const candidates: string[] = [];

  if (process.env.VIRTUAL_ENV) {
    candidates.push(
      resolve(process.env.VIRTUAL_ENV, 'bin/python'),
      resolve(process.env.VIRTUAL_ENV, 'bin/python3'),
    );
  }

  let currentDir = repoRoot;
  while (true) {
    candidates.push(
      resolve(currentDir, '.venv/bin/python'),
      resolve(currentDir, '.venv/bin/python3'),
    );
    const parentDir = resolve(currentDir, '..');
    if (parentDir === currentDir) {
      break;
    }
    currentDir = parentDir;
  }

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return 'python3';
};

export const buildBackendEnv = (
  project: RealBackendProjectEnv,
  overrides: Record<string, string> = {},
): NodeJS.ProcessEnv => ({
  ...process.env,
  ...getFixturePasswordEnv(),
  SERVER_LABEL: project.serverLabel,
  SERVER_PORT: String(project.apiPort),
  E2E_TEST_BASE_URL: project.apiBaseUrl,
  AUTH_MODE: project.authMode,
  TEST_MODE: 'true',
  DEFER_HEAVY_STARTUP: 'true',
  ENABLE_ADMIN_E2E_TEST_MODE: 'true',
  TLDW_ADMIN_E2E_SUPPORT_KEY:
    process.env.TLDW_ADMIN_E2E_SUPPORT_KEY || DEFAULT_ADMIN_E2E_SUPPORT_KEY,
  PYTEST_CURRENT_TEST: process.env.PYTEST_CURRENT_TEST || 'admin-ui-real-backend-e2e',
  JWT_ALGORITHM: process.env.JWT_ALGORITHM || 'HS256',
  JWT_SECRET_KEY: process.env.JWT_SECRET_KEY || 'playwright-test-secret-1234567890',
  SINGLE_USER_API_KEY: process.env.SINGLE_USER_API_KEY || 'single-user-admin-key',
  SINGLE_USER_TEST_API_KEY: process.env.SINGLE_USER_TEST_API_KEY || 'single-user-admin-key',
  ...(project.projectName === 'chromium-real-jwt' ? MANAGED_JWT_WEBHOOK_ENV : {}),
  DATABASE_URL:
    process.env.DATABASE_URL
    || `sqlite:////${tempRoot.replace(/^\/+/, '')}/${project.serverLabel}-authnz.db`,
  JOBS_DB_PATH:
    process.env.JOBS_DB_PATH
    || `${tempRoot}/${project.serverLabel}-jobs.db`,
  MONITORING_ALERTS_DB:
    process.env.MONITORING_ALERTS_DB
    || `${tempRoot}/${project.serverLabel}-monitoring-alerts.db`,
  TLDW_DB_BACKUP_PATH:
    process.env.TLDW_DB_BACKUP_PATH
    || `${tempRoot}/${project.serverLabel}-backups`,
  USER_DB_BASE_DIR:
    process.env.USER_DB_BASE_DIR
    || `${tempRoot}/${project.serverLabel}-userdbs`,
  ...overrides,
});

export const startManagedBackend = async (
  project: RealBackendProjectEnv,
  overrides: Record<string, string> = {},
): Promise<ManagedBackendProcess> => {
  if (!existsSync(lifecycleScript)) {
    throw new Error(`Server lifecycle script not found at ${lifecycleScript}`);
  }

  const command = getPythonCommand();
  const args = [lifecycleScript, 'start'];
  const childProcess = spawn(command, args, {
    cwd: repoRoot,
    env: buildBackendEnv(project, overrides),
    stdio: 'inherit',
  });

  await new Promise<void>((resolvePromise, reject) => {
    childProcess.once('spawn', () => resolvePromise());
    childProcess.once('error', reject);
  });

  return { process: childProcess, command, args };
};

const wait = async (ms: number): Promise<void> =>
  new Promise((resolvePromise) => {
    setTimeout(resolvePromise, ms);
  });

export const waitForManagedBackend = async (
  project: RealBackendProjectEnv,
  timeoutMs = 120_000,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  const candidates = [
    '/healthz',
    '/api/v1/healthz',
    '/readyz',
    '/api/v1/readyz',
    '/health',
    '/api/v1/health',
  ];

  while (Date.now() < deadline) {
    for (const path of candidates) {
      try {
        const response = await fetch(`${project.apiBaseUrl}${path}`);
        if (response.ok || (response.status === 206 && path.endsWith('/health'))) {
          return;
        }
      } catch {
        // Keep polling until timeout.
      }
    }
    await wait(2_000);
  }

  throw new Error(`Managed backend for ${project.projectName} did not become healthy within ${timeoutMs}ms`);
};

export const stopManagedBackend = async (
  project: RealBackendProjectEnv,
  overrides: Record<string, string> = {},
): Promise<void> => {
  if (!existsSync(lifecycleScript)) {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const command = getPythonCommand();
    const args = [lifecycleScript, 'stop'];
    const stopProcess = spawn(command, args, {
      cwd: repoRoot,
      env: buildBackendEnv(project, overrides),
      stdio: 'inherit',
    });

    stopProcess.once('error', reject);
    stopProcess.once('exit', (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`Failed to stop managed backend for ${project.projectName} (exit ${code ?? 'null'})`));
    });
  });
};
