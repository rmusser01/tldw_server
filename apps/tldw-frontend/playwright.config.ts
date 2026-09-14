import { defineConfig, devices } from '@playwright/test';

const rawBaseUrl = process.env.TLDW_WEB_URL || 'http://localhost:8080';
const baseURL = rawBaseUrl.replace('127.0.0.1', 'localhost');
const webCommand = process.env.TLDW_WEB_CMD || 'bun run dev -- -p 8080';
const shouldAutoStart = process.env.TLDW_WEB_AUTOSTART !== 'false';
const extensionPersistenceSuite = process.argv.some((arg) =>
  arg.includes('extension-api-key-persistence')
);
const cookieLifecycleSuite =
  process.env.TLDW_COOKIE_LIFECYCLE === '1' ||
  process.argv.some((arg) => arg.includes('single-user-cookie-lifecycle'));
const cookieLifecycleApiUrl =
  process.env.TLDW_COOKIE_LIFECYCLE_API_URL || 'http://127.0.0.1:18001';
const cookieLifecycleApiKey =
  process.env.TLDW_COOKIE_LIFECYCLE_API_KEY || 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY';
const rawApiUrl =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.TLDW_SERVER_URL ||
  process.env.TLDW_E2E_SERVER_URL ||
  'http://127.0.0.1:8000';
const defaultApiUrl = /^https?:\/\//i.test(rawApiUrl) ? rawApiUrl : `http://${rawApiUrl}`;
const webServerEnv = {
  ...Object.fromEntries(
    Object.entries(process.env).filter(
      (entry): entry is [string, string] => typeof entry[1] === 'string'
    )
  ),
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: cookieLifecycleSuite
    ? 'quickstart'
    : process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || 'advanced',
  NEXT_PUBLIC_API_URL: cookieLifecycleSuite ? '' : defaultApiUrl,
  ...(cookieLifecycleSuite
    ? {
        AUTH_MODE: 'single_user',
        SINGLE_USER_API_KEY: cookieLifecycleApiKey,
        TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: '1',
        SINGLE_USER_SESSION_COOKIE_NAME: 'tldw_single_user_session',
        TLDW_INTERNAL_API_ORIGIN: cookieLifecycleApiUrl,
      }
    : {}),
};

export default defineConfig({
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: shouldAutoStart && !extensionPersistenceSuite
    ? {
        command: webCommand,
        env: webServerEnv,
        url: baseURL,
        reuseExistingServer: true,
        timeout: 120_000,
      }
    : undefined,
  projects: [
    {
      name: 'chromium',
      testDir: 'e2e',
      testIgnore: ['**/workflows/tier-*/**', '**/workflows/journeys/**'],
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'standalone-html-firefox',
      testDir: 'e2e/workflows',
      testMatch: 'presentation-studio-standalone-html.security.spec.ts',
      retries: 0,
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'standalone-html-webkit',
      testDir: 'e2e/workflows',
      testMatch: 'presentation-studio-standalone-html.security.spec.ts',
      retries: 0,
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'tier-1',
      testDir: 'e2e/workflows/tier-1-critical',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'tier-2',
      testDir: 'e2e/workflows/tier-2-features',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'tier-3',
      testDir: 'e2e/workflows/tier-3-automation',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'tier-4',
      testDir: 'e2e/workflows/tier-4-admin',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'tier-5',
      testDir: 'e2e/workflows/tier-5-specialized',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'journeys',
      testDir: 'e2e/workflows/journeys',
      timeout: 120_000,
      expect: { timeout: 30_000 },
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
