import { defineConfig, devices } from '@playwright/test';

import { getProjectEnv } from './tests/e2e/real-backend/helpers/project-env';

const rawBaseUrl = process.env.TLDW_ADMIN_UI_URL || 'http://127.0.0.1:3001';
const baseURL = rawBaseUrl.replace('localhost', '127.0.0.1');
const webCommand = process.env.TLDW_ADMIN_UI_CMD || 'bun run dev -- --hostname 127.0.0.1';
const shouldAutoStart = process.env.TLDW_ADMIN_UI_AUTOSTART !== 'false';
const shouldStartRealBackendUiServers = process.argv.some((arg) => (
  arg.includes('real-backend') || arg.includes('chromium-real-')
));

const realJwtProject = getProjectEnv('chromium-real-jwt');
const realSingleUserProject = getProjectEnv('chromium-real-single-user');

const baseUiEnv = {
  ...process.env,
  ADMIN_UI_ALLOW_API_KEY_LOGIN: process.env.ADMIN_UI_ALLOW_API_KEY_LOGIN || 'true',
  JWT_ALGORITHM: process.env.JWT_ALGORITHM || 'HS256',
  JWT_SECRET_KEY: process.env.JWT_SECRET_KEY || 'playwright-test-secret-1234567890',
  NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN: process.env.NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN || 'true',
  NEXT_TELEMETRY_DISABLED: '1',
};

const realBackendUiServers = shouldAutoStart && shouldStartRealBackendUiServers
  ? [
      {
        command: `bunx next dev -p ${realJwtProject.uiPort} --hostname 127.0.0.1`,
        url: realJwtProject.uiBaseUrl,
        // These auth-mode-specific UI servers must boot with the exact backend URL
        // and auth env for the current project; reusing a stale process leaks config.
        reuseExistingServer: false,
        timeout: 120_000,
        env: {
          ...baseUiEnv,
          AUTH_MODE: 'multi_user',
          NEXT_PUBLIC_API_URL: realJwtProject.apiBaseUrl,
          TLDW_ADMIN_E2E_REAL_BACKEND: 'true',
          TEST_MODE: 'true',
        },
      },
      {
        command: `bunx next dev -p ${realSingleUserProject.uiPort} --hostname 127.0.0.1`,
        url: realSingleUserProject.uiBaseUrl,
        // These auth-mode-specific UI servers must boot with the exact backend URL
        // and auth env for the current project; reusing a stale process leaks config.
        reuseExistingServer: false,
        timeout: 120_000,
        env: {
          ...baseUiEnv,
          AUTH_MODE: 'single_user',
          NEXT_PUBLIC_API_URL: realSingleUserProject.apiBaseUrl,
          SINGLE_USER_API_KEY: process.env.SINGLE_USER_API_KEY || 'single-user-admin-key',
          TLDW_ADMIN_E2E_REAL_BACKEND: 'true',
          TEST_MODE: 'true',
        },
      },
    ]
  : [];

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 60_000,
  globalSetup: './tests/e2e/real-backend/helpers/global-setup.ts',
  globalTeardown: './tests/e2e/real-backend/helpers/global-teardown.ts',
  expect: {
    timeout: 15_000,
  },
  retries: process.env.CI ? 2 : 0,
  reporter: 'line',
  outputDir: '/tmp/tldw-admin-playwright',
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: shouldAutoStart
    ? [
        {
          command: webCommand,
          url: baseURL,
          reuseExistingServer: true,
          timeout: 120_000,
          env: {
            ...baseUiEnv,
            AUTH_MODE: process.env.AUTH_MODE || 'single_user',
            NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5999',
          },
        },
        ...realBackendUiServers,
      ]
    : undefined,
  projects: [
    {
      name: 'chromium',
      testIgnore: ['tests/e2e/real-backend/**'],
      use: { ...devices['Desktop Chrome'], baseURL },
    },
    {
      name: 'chromium-real-jwt',
      testMatch: ['tests/e2e/real-backend/**/*.spec.ts'],
      use: { ...devices['Desktop Chrome'], baseURL: realJwtProject.uiBaseUrl },
    },
    {
      name: 'chromium-real-single-user',
      testMatch: ['tests/e2e/real-backend/**/*.spec.ts'],
      use: { ...devices['Desktop Chrome'], baseURL: realSingleUserProject.uiBaseUrl },
    },
  ],
});
