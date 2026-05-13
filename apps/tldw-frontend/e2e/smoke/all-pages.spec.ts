/**
 * Smoke tests for all pages in tldw-frontend
 *
 * These tests visit each page and verify:
 * - Page loads without error boundaries
 * - No uncaught JavaScript errors
 * - No critical console errors
 * - Network requests complete (or fail gracefully)
 *
 * Run with: npm run e2e:smoke
 */

import type { Page } from '@playwright/test';
import type { DiagnosticsData } from './smoke.setup';
import {
  test,
  expect,
  seedAuth,
  getCriticalIssues,
  classifySmokeIssues,
} from './smoke.setup';
import { waitForAppShell } from '../utils/helpers';
import { PAGES, PageEntry, getActivePages, PAGE_COUNT, ACTIVE_PAGE_COUNT } from './page-inventory';

// Test configuration
const LOAD_TIMEOUT = 30_000; // 30s max for page load
const ELEMENT_TIMEOUT = 15_000; // 15s max for element visibility
const NETWORK_IDLE_TIMEOUT = Math.max(
  1_000,
  Number(process.env.TLDW_SMOKE_NETWORK_IDLE_TIMEOUT_MS || 8_000)
);
const VERBOSE_CONSOLE = process.env.TLDW_SMOKE_VERBOSE_CONSOLE === '1';
const ALLOWLIST_LOG_LIMIT = Number(process.env.TLDW_SMOKE_ALLOWLIST_LOG_LIMIT || 10);
const SMOKE_HARD_GATE = process.env.TLDW_SMOKE_HARD_GATE !== '0';
const TRANSIENT_RUNTIME_RETRY_ATTEMPTS = Math.max(
  0,
  Number(process.env.TLDW_SMOKE_TRANSIENT_RUNTIME_RETRIES || 2)
);
const TRANSIENT_NAVIGATION_RETRY_ATTEMPTS = Math.max(
  0,
  Number(process.env.TLDW_SMOKE_TRANSIENT_NAVIGATION_RETRIES || 1)
);
const MAX_TRANSIENT_RETRY_ATTEMPTS = Math.max(
  TRANSIENT_RUNTIME_RETRY_ATTEMPTS,
  TRANSIENT_NAVIGATION_RETRY_ATTEMPTS
);
const TRANSIENT_RUNTIME_RETRY_DELAY_MS = Math.max(
  0,
  Number(process.env.TLDW_SMOKE_TRANSIENT_RUNTIME_RETRY_DELAY_MS || 500)
);
const ROUTE_TEST_TIMEOUT = Math.max(
  60_000,
  LOAD_TIMEOUT * (MAX_TRANSIENT_RETRY_ATTEMPTS + 1) + ELEMENT_TIMEOUT + 15_000
);
const TRANSIENT_NAVIGATION_TIMEOUT_PATTERN = /page\.goto: Timeout/i;
const KEY_NAV_TARGETS = ['/chat', '/media', '/knowledge', '/notes', '/prompts', '/settings/tldw'];
const WAYFINDING_404_PATH = '/__wayfinding-missing-route__';
const ROUTE_ERROR_FIXTURE_QUERY_KEY = '__forceRouteError';
const NON_CORE_ROUTE_BOUNDARY_TARGETS = [
  { name: 'Server Admin', path: '/admin/server', routeId: 'admin-server', routeLabel: 'Server Admin' },
  {
    name: 'Llama.cpp Admin',
    path: '/admin/llamacpp',
    routeId: 'admin-llamacpp',
    routeLabel: 'Llama.cpp Admin',
  },
  { name: 'MLX Admin', path: '/admin/mlx', routeId: 'admin-mlx', routeLabel: 'MLX Admin' },
  {
    name: 'Content Review',
    path: '/content-review',
    routeId: 'content-review',
    routeLabel: 'Content Review',
  },
  { name: 'Data Tables', path: '/data-tables', routeId: 'data-tables', routeLabel: 'Data Tables' },
  {
    name: 'Kanban Playground',
    path: '/kanban',
    routeId: 'kanban-playground',
    routeLabel: 'Kanban Playground',
  },
  {
    name: 'Chunking Playground',
    path: '/chunking-playground',
    routeId: 'chunking-playground',
    routeLabel: 'Chunking Playground',
  },
  {
    name: 'Moderation Review',
    path: '/moderation',
    routeId: 'moderation-review',
    routeLabel: 'Moderation Review',
  },
  {
    name: 'Content Rules',
    path: '/moderation/rules',
    routeId: 'moderation-rules',
    routeLabel: 'Content Rules',
  },
  { name: 'Collections', path: '/collections', routeId: 'collections', routeLabel: 'Collections' },
  {
    name: 'World Books',
    path: '/world-books',
    routeId: 'world-books',
    routeLabel: 'World Books',
  },
  {
    name: 'Dictionaries',
    path: '/dictionaries',
    routeId: 'dictionaries',
    routeLabel: 'Dictionaries',
  },
  { name: 'Characters', path: '/characters', routeId: 'characters', routeLabel: 'Characters' },
  { name: 'Items', path: '/items', routeId: 'items', routeLabel: 'Items' },
  {
    name: 'Document Workspace',
    path: '/document-workspace',
    routeId: 'document-workspace',
    routeLabel: 'Document Workspace',
  },
  {
    name: 'Speech Playground',
    path: '/speech',
    routeId: 'speech',
    routeLabel: 'Speech Playground',
  },
] as const;
const RUNTIME_OVERLAY_PATTERNS = [
  /Runtime(?:\s+\w+)?\s+Error/i,
  /Runtime SyntaxError/i,
  /Invalid or unexpected token/i,
  /Objects are not valid as a React child/i,
  /message\.error is not a function/i,
];
const TRANSIENT_RUNTIME_OVERLAY_PATTERNS = [
  /Runtime SyntaxError/i,
  /Invalid or unexpected token/i,
  /Unexpected end of input/i,
];

const keyNavEntries: PageEntry[] = KEY_NAV_TARGETS.map((targetPath) =>
  PAGES.find((entry) => entry.path === targetPath)
).filter((entry): entry is PageEntry => Boolean(entry));

/**
 * Format diagnostics for console output
 */
function formatDiagnostics(
  entry: PageEntry,
  issues: ReturnType<typeof getCriticalIssues>,
  classifiedIssues: ReturnType<typeof classifySmokeIssues>
): string {
  const lines: string[] = [];

  if (issues.pageErrors.length) {
    lines.push(`  PAGE ERRORS (${issues.pageErrors.length}):`);
    issues.pageErrors.forEach((e) => {
      lines.push(`    - ${e.message}`);
      if (e.stack) {
        const firstStackLine = e.stack.split('\n')[1]?.trim();
        if (firstStackLine) lines.push(`      ${firstStackLine}`);
      }
    });
  }

  if (classifiedIssues.allowlistedConsoleErrors.length) {
    lines.push(`  ALLOWLISTED CONSOLE WARNINGS (${classifiedIssues.allowlistedConsoleErrors.length}):`);
    const loggedEntries = classifiedIssues.allowlistedConsoleErrors.slice(0, ALLOWLIST_LOG_LIMIT);
    loggedEntries.forEach(({ entry: c, rule }) => {
      const text = VERBOSE_CONSOLE
        ? c.text
        : c.text.length > 200
          ? c.text.slice(0, 200) + '...'
          : c.text;
      lines.push(`    - [${rule.id}] ${text}`);
    });
    const omitted = classifiedIssues.allowlistedConsoleErrors.length - loggedEntries.length;
    if (omitted > 0) {
      lines.push(`    - ... ${omitted} additional allowlisted console warnings omitted`);
    }
  }

  if (classifiedIssues.unexpectedConsoleErrors.length) {
    lines.push(`  UNEXPECTED CONSOLE ERRORS (${classifiedIssues.unexpectedConsoleErrors.length}):`);
    classifiedIssues.unexpectedConsoleErrors.forEach((c) => {
      const text = VERBOSE_CONSOLE
        ? c.text
        : c.text.length > 200
          ? c.text.slice(0, 200) + '...'
          : c.text;
      lines.push(`    - ${text}`);
    });
  }

  if (classifiedIssues.allowlistedRequestFailures.length) {
    lines.push(`  ALLOWLISTED REQUEST FAILURES (${classifiedIssues.allowlistedRequestFailures.length}):`);
    const loggedEntries = classifiedIssues.allowlistedRequestFailures.slice(0, ALLOWLIST_LOG_LIMIT);
    loggedEntries.forEach(({ entry: r, rule }) => {
      lines.push(`    - [${rule.id}] ${r.url} (${r.errorText})`);
    });
    const omitted = classifiedIssues.allowlistedRequestFailures.length - loggedEntries.length;
    if (omitted > 0) {
      lines.push(`    - ... ${omitted} additional allowlisted request failures omitted`);
    }
  }

  if (classifiedIssues.unexpectedRequestFailures.length) {
    lines.push(`  UNEXPECTED REQUEST FAILURES (${classifiedIssues.unexpectedRequestFailures.length}):`);
    classifiedIssues.unexpectedRequestFailures.forEach((r) => {
      lines.push(`    - ${r.url} (${r.errorText})`);
    });
  }

  return lines.length ? `\n${entry.path}:\n${lines.join('\n')}` : '';
}

function assertWarningHardGate(
  routePath: string,
  classifiedIssues: ReturnType<typeof classifySmokeIssues>
): void {
  if (!SMOKE_HARD_GATE) {
    return;
  }

  expect(
    classifiedIssues.unexpectedConsoleErrors,
    `Unexpected console errors on ${routePath}: ${classifiedIssues.unexpectedConsoleErrors
      .map((entry) => entry.text)
      .join(' | ')}`
  ).toHaveLength(0);

  expect(
    classifiedIssues.unexpectedRequestFailures,
    `Unexpected request failures on ${routePath}: ${classifiedIssues.unexpectedRequestFailures
      .map((entry) => `${entry.url} (${entry.errorText})`)
      .join(' | ')}`
  ).toHaveLength(0);
}

function hasRuntimeOverlaySignal(input: string): boolean {
  return RUNTIME_OVERLAY_PATTERNS.some((pattern) => pattern.test(input));
}

function hasTransientRuntimeOverlaySignal(input: string): boolean {
  return TRANSIENT_RUNTIME_OVERLAY_PATTERNS.some((pattern) => pattern.test(input));
}

function resetDiagnostics(diagnostics: DiagnosticsData): void {
  diagnostics.console.length = 0;
  diagnostics.pageErrors.length = 0;
  diagnostics.requestFailures.length = 0;
}

async function waitForTransientRetryReady(page: Page): Promise<void> {
  const retryWindowMs = Math.max(1_000, TRANSIENT_RUNTIME_RETRY_DELAY_MS);
  await expect
    .poll(
      async () => {
        const bodyText = await page
          .evaluate(() => document.body?.innerText ?? '')
          .catch(() => '');
        const shellVisible = await page
          .locator('#root, #__next')
          .first()
          .isVisible()
          .catch(() => false);
        return shellVisible && !hasTransientRuntimeOverlaySignal(bodyText);
      },
      { timeout: retryWindowMs }
    )
    .toBe(true)
    .catch(() => {});
}

type RouteVisitResult = {
  response: Awaited<ReturnType<Page['goto']>>;
  issues: ReturnType<typeof getCriticalIssues>;
  classifiedIssues: ReturnType<typeof classifySmokeIssues>;
  retriesUsed: number;
};

async function visitRouteWithTransientRetry(
  page: Page,
  diagnostics: DiagnosticsData,
  routePath: string
): Promise<RouteVisitResult> {
  let retriesUsed = 0;
  let response: Awaited<ReturnType<Page['goto']>> = null;
  let issues: ReturnType<typeof getCriticalIssues> = {
    pageErrors: [],
    consoleErrors: [],
    requestFailures: [],
  };
  let classifiedIssues: ReturnType<typeof classifySmokeIssues> = {
    pageErrors: [],
    allowlistedConsoleErrors: [],
    unexpectedConsoleErrors: [],
    allowlistedRequestFailures: [],
    unexpectedRequestFailures: [],
  };

  for (let attempt = 0; attempt <= MAX_TRANSIENT_RETRY_ATTEMPTS; attempt += 1) {
    resetDiagnostics(diagnostics);
    try {
      response = await page.goto(routePath, {
        waitUntil: 'domcontentloaded',
        timeout: LOAD_TIMEOUT,
      });
      await waitForAppShell(page, NETWORK_IDLE_TIMEOUT);
    } catch (error) {
      const errorMessage = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      const isNavigationTimeout = TRANSIENT_NAVIGATION_TIMEOUT_PATTERN.test(errorMessage);
      const shouldRetryNavigationTimeout =
        isNavigationTimeout && attempt < TRANSIENT_NAVIGATION_RETRY_ATTEMPTS;
      if (!shouldRetryNavigationTimeout) {
        throw error;
      }

      retriesUsed = attempt + 1;
      console.warn(
        `[smoke-transient-navigation-retry] ${routePath} retry ${retriesUsed}/${TRANSIENT_NAVIGATION_RETRY_ATTEMPTS} after navigation timeout`
      );
      await waitForTransientRetryReady(page);
      continue;
    }

    issues = getCriticalIssues(diagnostics);
    classifiedIssues = classifySmokeIssues(routePath, issues);

    const bodyText = await page.evaluate(() => document.body?.innerText ?? '').catch(() => '');
    const transientSignals = [
      ...issues.pageErrors.map((entry) => entry.message).filter(hasTransientRuntimeOverlaySignal),
      ...issues.consoleErrors.map((entry) => entry.text).filter(hasTransientRuntimeOverlaySignal),
    ];
    if (hasTransientRuntimeOverlaySignal(bodyText)) {
      transientSignals.push('runtime-overlay-body-signal');
    }

    const shouldRetry =
      transientSignals.length > 0 && attempt < TRANSIENT_RUNTIME_RETRY_ATTEMPTS;
    if (!shouldRetry) {
      break;
    }

    retriesUsed = attempt + 1;
    console.warn(
      `[smoke-transient-runtime-retry] ${routePath} retry ${retriesUsed}/${TRANSIENT_RUNTIME_RETRY_ATTEMPTS} after transient runtime overlay signal: ${transientSignals[0]}`
    );
    await waitForTransientRetryReady(page);
  }

  return {
    response,
    issues,
    classifiedIssues,
    retriesUsed,
  };
}

async function assertNoRuntimeOverlay(
  page: Page,
  issues: ReturnType<typeof getCriticalIssues>,
  context: string
): Promise<void> {
  const runtimeConsoleErrors = issues.consoleErrors
    .map((entry) => entry.text)
    .filter(hasRuntimeOverlaySignal);

  const overlaySnapshot = await page.evaluate(() => ({
    bodyText: document.body?.innerText ?? '',
  }));

  const bodyHasRuntimeSignal = hasRuntimeOverlaySignal(overlaySnapshot.bodyText);
  const bodySnippet = bodyHasRuntimeSignal
    ? overlaySnapshot.bodyText.replace(/\s+/g, ' ').trim().slice(0, 220)
    : '';
  const hasRuntimeOverlay = runtimeConsoleErrors.length > 0 || bodyHasRuntimeSignal;

  expect(
    hasRuntimeOverlay,
    [
      `Runtime overlay detected on ${context}.`,
      `Console matches: ${runtimeConsoleErrors.length}`,
      bodySnippet ? `Body snippet: ${bodySnippet}` : '',
    ]
      .filter(Boolean)
      .join(' '),
  ).toBeFalsy();
}

test.describe('Smoke Tests - All Pages', () => {
  test.describe.configure({ mode: 'parallel', retries: 0 });

  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  // Log test suite info
  test.beforeAll(() => {
    console.log(
      `\nSmoke test suite: ${ACTIVE_PAGE_COUNT} pages (${PAGE_COUNT - ACTIVE_PAGE_COUNT} skipped)\n`
    );
  });

  // Generate a test for each active page
  for (const entry of getActivePages()) {
    test(`${entry.name} (${entry.path})`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      // Navigate to the page with transient retry handling for dev-runtime flakes.
      const { response, issues, classifiedIssues, retriesUsed } = await visitRouteWithTransientRetry(
        page,
        diagnostics,
        entry.path
      );

      // Check HTTP response status
      const status = response?.status();
      if (status && status >= 400 && status !== 404) {
        // 404 is handled separately, other 4xx/5xx are issues
        console.warn(`HTTP ${status} for ${entry.path}`);
      }

      // Check for error boundary UI patterns
      const errorBoundaryVisible = await page
        .getByTestId('error-boundary')
        .first()
        .isVisible()
        .catch(() => false);

      const errorTextVisible = await page
        .getByText(/something went wrong/i)
        .first()
        .isVisible()
        .catch(() => false);

      // If page has an expected test ID, verify it's visible
      if (entry.expectedTestId) {
        await expect(page.getByTestId(entry.expectedTestId)).toBeVisible({
          timeout: ELEMENT_TIMEOUT,
        });
      }
      await assertNoRuntimeOverlay(page, issues, entry.path);

      // Log any issues found (useful for debugging)
      const diagnosticOutput = formatDiagnostics(entry, issues, classifiedIssues);
      if (diagnosticOutput) {
        console.log(diagnosticOutput);
      }
      if (retriesUsed > 0) {
        console.log(
          `[smoke-transient-runtime-retry] ${entry.path} stabilized after ${retriesUsed} retr${retriesUsed === 1 ? 'y' : 'ies'}`
        );
      }

      // Assertions
      expect(
        errorBoundaryVisible,
        `Error boundary [data-testid="error-boundary"] triggered on ${entry.path}`
      ).toBeFalsy();

      expect(errorTextVisible, `"Something went wrong" text visible on ${entry.path}`).toBeFalsy();

      // Page errors are hard failures
      expect(
        issues.pageErrors,
        `Uncaught page errors on ${entry.path}: ${issues.pageErrors.map((e) => e.message).join(', ')}`
      ).toHaveLength(0);
      assertWarningHardGate(entry.path, classifiedIssues);
    });
  }
});

// Category-specific test suites for selective running
test.describe('Smoke Tests - Chat', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const entry of PAGES.filter((p) => p.category === 'chat' && !p.skip)) {
    test(`${entry.name}`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const { issues, classifiedIssues } = await visitRouteWithTransientRetry(
        page,
        diagnostics,
        entry.path
      );
      assertWarningHardGate(entry.path, classifiedIssues);
      expect(issues.pageErrors).toHaveLength(0);
    });
  }
});

test.describe('Smoke Tests - Settings', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const entry of PAGES.filter((p) => p.category === 'settings' && !p.skip)) {
    test(`${entry.name}`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const { issues, classifiedIssues } = await visitRouteWithTransientRetry(
        page,
        diagnostics,
        entry.path
      );
      assertWarningHardGate(entry.path, classifiedIssues);
      expect(issues.pageErrors).toHaveLength(0);
    });
  }
});

test.describe('Smoke Tests - Admin', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const entry of PAGES.filter((p) => p.category === 'admin' && !p.skip)) {
    test(`${entry.name}`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const { issues, classifiedIssues } = await visitRouteWithTransientRetry(
        page,
        diagnostics,
        entry.path
      );
      assertWarningHardGate(entry.path, classifiedIssues);
      expect(issues.pageErrors).toHaveLength(0);
    });
  }
});

test.describe('Smoke Tests - Workspace', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const entry of PAGES.filter((p) => p.category === 'workspace' && !p.skip)) {
    test(`${entry.name}`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const { issues, classifiedIssues } = await visitRouteWithTransientRetry(
        page,
        diagnostics,
        entry.path
      );
      assertWarningHardGate(entry.path, classifiedIssues);
      expect(issues.pageErrors).toHaveLength(0);
    });
  }
});

test.describe('Smoke Tests - Knowledge', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const entry of PAGES.filter((p) => p.category === 'knowledge' && !p.skip)) {
    test(`${entry.name}`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const { issues, classifiedIssues } = await visitRouteWithTransientRetry(
        page,
        diagnostics,
        entry.path
      );
      assertWarningHardGate(entry.path, classifiedIssues);
      expect(issues.pageErrors).toHaveLength(0);
    });
  }
});

test.describe('Smoke Tests - Audio', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const entry of PAGES.filter((p) => p.category === 'audio' && !p.skip)) {
    test(`${entry.name}`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const { issues } = await visitRouteWithTransientRetry(page, diagnostics, entry.path);
      expect(issues.pageErrors).toHaveLength(0);
    });
  }
});

test.describe('Smoke Tests - Key Navigation Targets', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  test('inventory contains all key nav routes', () => {
    expect(keyNavEntries).toHaveLength(KEY_NAV_TARGETS.length);
  });

  for (const entry of keyNavEntries) {
    test(`${entry.name} (${entry.path})`, async ({ page, diagnostics }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const response = await page.goto(entry.path, {
        waitUntil: 'domcontentloaded',
        timeout: LOAD_TIMEOUT,
      });
      await waitForAppShell(page, NETWORK_IDLE_TIMEOUT);

      const issues = getCriticalIssues(diagnostics);
      const classifiedIssues = classifySmokeIssues(entry.path, issues);
      const status = response?.status() ?? 0;
      await assertNoRuntimeOverlay(page, issues, `key-nav:${entry.path}`);

      expect(
        status === 0 || status < 400,
        `Expected key target ${entry.path} to load successfully (status: ${status})`
      ).toBeTruthy();
      expect(
        issues.pageErrors,
        `Uncaught page errors on key nav target ${entry.path}`
      ).toHaveLength(0);
      assertWarningHardGate(entry.path, classifiedIssues);
    });
  }
});

test.describe('Smoke Tests - Wayfinding', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  test('settings route shows active location context', async ({ page, diagnostics }) => {
    test.setTimeout(ROUTE_TEST_TIMEOUT);
    const response = await page.goto('/settings/tldw', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT,
    });
    await waitForAppShell(page, NETWORK_IDLE_TIMEOUT);

    const status = response?.status() ?? 0;
    test.skip(
      status >= 400,
      `Settings wayfinding markers unavailable in this runtime (status: ${status})`
    );

    await expect(page.getByTestId('settings-navigation')).toBeVisible();
    await expect(page.getByTestId('settings-current-section')).toBeVisible();

    const activeSettingsLink = page.locator(
      '[data-testid^="settings-nav-link-"][aria-current="page"]'
    );
    await expect(activeSettingsLink.first()).toBeVisible();

    const issues = getCriticalIssues(diagnostics);
    const classifiedIssues = classifySmokeIssues('/settings/tldw', issues);
    await assertNoRuntimeOverlay(page, issues, 'wayfinding:/settings/tldw');
    expect(
      issues.pageErrors,
      'Uncaught page errors while validating settings wayfinding'
    ).toHaveLength(0);
    assertWarningHardGate('/settings/tldw', classifiedIssues);
  });

  test('legacy alias redirects to canonical destination with params preserved', async ({
    page,
    diagnostics,
  }) => {
    test.setTimeout(ROUTE_TEST_TIMEOUT);
    const response = await page.goto('/search?q=wayfinding-smoke', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT,
    });
    const status = response?.status() ?? 0;
    test.skip(status >= 400, `Alias redirect path unavailable in this runtime (status: ${status})`);

    await page.waitForURL(/\/knowledge\?q=wayfinding-smoke/, {
      timeout: LOAD_TIMEOUT,
    });
    await waitForAppShell(page, NETWORK_IDLE_TIMEOUT);

    const issues = getCriticalIssues(diagnostics);
    const classifiedIssues = classifySmokeIssues('/knowledge', issues);
    await assertNoRuntimeOverlay(page, issues, 'wayfinding:/search -> /knowledge');
    expect(
      issues.pageErrors,
      'Uncaught page errors while validating alias redirect wayfinding'
    ).toHaveLength(0);
    assertWarningHardGate('/knowledge', classifiedIssues);
  });

  test('404 recovery controls keep predictable keyboard order', async ({ page, diagnostics }) => {
    test.setTimeout(ROUTE_TEST_TIMEOUT);
    await page.goto(WAYFINDING_404_PATH, {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT,
    });
    await waitForAppShell(page, NETWORK_IDLE_TIMEOUT);

    const hasWayfindingPanel = (await page.getByTestId('not-found-recovery-panel').count()) > 0;
    test.skip(
      !hasWayfindingPanel,
      'Wayfinding-specific 404 recovery panel not available in this runtime target'
    );

    await expect(page.getByRole('heading', { name: 'We could not find that route' })).toBeVisible();
    await expect(page.getByTestId('not-found-recovery-panel')).toBeVisible();

    const controlOrder = await page
      .locator("[data-testid='not-found-recovery-panel'] [data-testid]")
      .evaluateAll((elements) => elements.map((element) => element.getAttribute('data-testid')));

    expect(controlOrder).toEqual([
      'not-found-go-chat',
      'not-found-open-research',
      'not-found-open-media',
      'not-found-open-settings',
      'not-found-go-back',
    ]);

    const goChatButton = page.getByTestId('not-found-go-chat');
    await goChatButton.focus();
    await expect(goChatButton).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(page.getByTestId('not-found-open-research')).toBeFocused();

    const issues = getCriticalIssues(diagnostics);
    const classifiedIssues = classifySmokeIssues(WAYFINDING_404_PATH, issues);
    await assertNoRuntimeOverlay(page, issues, `wayfinding:${WAYFINDING_404_PATH}`);
    expect(
      issues.pageErrors,
      'Uncaught page errors while validating 404 recovery wayfinding'
    ).toHaveLength(0);
    assertWarningHardGate(WAYFINDING_404_PATH, classifiedIssues);
  });
});

test.describe('Smoke Tests - Route Error Boundaries', () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
  });

  for (const target of NON_CORE_ROUTE_BOUNDARY_TARGETS) {
    test(`${target.name} forced-error fixture renders recovery contract`, async ({
      page,
      diagnostics,
    }) => {
      test.setTimeout(ROUTE_TEST_TIMEOUT);
      const fixturePath = `${target.path}?${ROUTE_ERROR_FIXTURE_QUERY_KEY}=${target.routeId}`;
      const response = await page.goto(fixturePath, {
        waitUntil: 'domcontentloaded',
        timeout: LOAD_TIMEOUT,
      });
      const status = response?.status() ?? 0;
      test.skip(
        status >= 400,
        `Route boundary fixture unavailable for ${target.path} in this runtime (status: ${status})`
      );

      await waitForAppShell(page, NETWORK_IDLE_TIMEOUT);
      await expect(page.getByTestId('error-boundary')).toBeVisible();
      await expect(page.getByTestId(`route-error-boundary-${target.routeId}`)).toBeVisible();
      await expect(page.getByTestId('route-error-retry')).toBeVisible();
      await expect(page.getByTestId('route-error-go-chat')).toBeVisible();
      await expect(page.getByTestId('route-error-open-settings')).toBeVisible();
      await expect(page.getByTestId('route-error-reload')).toBeVisible();
      await expect(page.getByTestId('route-error-route-label')).toHaveText(target.routeLabel);

      const issues = getCriticalIssues(diagnostics);
      const classifiedIssues = classifySmokeIssues(target.path, issues);
      await assertNoRuntimeOverlay(page, issues, `route-boundary:${fixturePath}`);
      assertWarningHardGate(target.path, classifiedIssues);
    });
  }
});
