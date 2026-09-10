import { describe, expect, it } from 'vitest';

import {
  getProjectEnv,
  getRequestedRealBackendProjects,
  shouldManageBackend,
} from '@/tests/e2e/real-backend/helpers/project-env';

describe('real-backend project env', () => {
  it('returns the default real-backend project urls when no overrides are present', () => {
    const project = getProjectEnv('chromium-real-jwt', {});

    expect(project.uiBaseUrl).toBe('http://127.0.0.1:3101');
    expect(project.apiBaseUrl).toBe('http://127.0.0.1:8101');
    expect(shouldManageBackend(project.projectName, {})).toBe(true);
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
