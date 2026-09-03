import { afterEach, describe, expect, it, vi } from 'vitest';

import { buildBackendEnv } from '@/tests/e2e/real-backend/helpers/backend-lifecycle';
import {
  getProjectEnv,
  getRequestedRealBackendProjects,
  shouldManageBackend,
} from '@/tests/e2e/real-backend/helpers/project-env';

describe('real-backend project env', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('returns the default real-backend project urls when no overrides are present', () => {
    const project = getProjectEnv('chromium-real-jwt', {});

    expect(project.uiBaseUrl).toBe('http://127.0.0.1:3101');
    expect(project.apiBaseUrl).toBe('http://127.0.0.1:8101');
    expect(shouldManageBackend(project.projectName, {})).toBe(true);
  });

  it('enables the canonical webhook runtime for the managed jwt project', () => {
    for (const name of [
      'TLDW_ADMIN_WEBHOOKS_MODE',
      'TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV',
      'TLDW_ADMIN_WEBHOOKS_E2E_LOOPBACK',
      'TLDW_ADMIN_WEBHOOK_KEYS_JSON',
      'TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID',
      'TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS',
      'TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_FRESHNESS_SECONDS',
    ]) {
      vi.stubEnv(name, 'ambient-value');
    }

    const env = buildBackendEnv(getProjectEnv('chromium-real-jwt', {}));

    expect(env).toMatchObject({
      TLDW_ADMIN_WEBHOOKS_MODE: 'on',
      TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV: 'true',
      TLDW_ADMIN_WEBHOOKS_E2E_LOOPBACK: 'true',
      TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID: 'admin-e2e-primary',
      TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS: '1',
      TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_FRESHNESS_SECONDS: '10',
    });
    expect(JSON.parse(env.TLDW_ADMIN_WEBHOOK_KEYS_JSON ?? '{}')).toEqual({
      'admin-e2e-primary': 'd3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3c=',
    });
  });

  it('keeps explicit managed jwt webhook overrides authoritative', () => {
    const env = buildBackendEnv(
      getProjectEnv('chromium-real-jwt', {}),
      {
        TLDW_ADMIN_WEBHOOKS_MODE: 'migrate',
        TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS: '7',
      },
    );

    expect(env.TLDW_ADMIN_WEBHOOKS_MODE).toBe('migrate');
    expect(env.TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS).toBe('7');
    expect(env.TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID).toBe('admin-e2e-primary');
  });

  it('does not inject managed jwt webhook defaults into the single-user project', () => {
    vi.stubEnv('TLDW_ADMIN_WEBHOOKS_MODE', 'ambient-mode');
    vi.stubEnv('TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID', 'ambient-primary');

    const env = buildBackendEnv(getProjectEnv('chromium-real-single-user', {}));

    expect(env.TLDW_ADMIN_WEBHOOKS_MODE).toBe('ambient-mode');
    expect(env.TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID).toBe('ambient-primary');
  });

  it('uses explicit backend url overrides for the jwt project and stops managing that backend', () => {
    const env = {
      TLDW_ADMIN_E2E_JWT_API_URL: 'http://127.0.0.1:9101',
    };

    const project = getProjectEnv('chromium-real-jwt', env);

    expect(project.apiBaseUrl).toBe('http://127.0.0.1:9101');
    expect(shouldManageBackend(project.projectName, env)).toBe(false);
  });

  it('disables managed backend startup globally when autostart is turned off', () => {
    const env = { TLDW_ADMIN_E2E_AUTOSTART_BACKEND: 'false' };

    expect(shouldManageBackend('chromium-real-jwt', env)).toBe(false);
    expect(shouldManageBackend('chromium-real-single-user', env)).toBe(false);
  });

  it('selects an equals-delimited real-backend Playwright project', () => {
    expect(
      getRequestedRealBackendProjects([
        'node',
        'playwright',
        'test',
        '--project=chromium-real-jwt',
      ]),
    ).toEqual(['chromium-real-jwt']);
  });

  it('selects a space-delimited real-backend project and ignores generic projects', () => {
    expect(
      getRequestedRealBackendProjects([
        'node',
        'playwright',
        'test',
        '--project',
        'chromium',
        '--project',
        'chromium-real-single-user',
      ]),
    ).toEqual(['chromium-real-single-user']);
  });
});
