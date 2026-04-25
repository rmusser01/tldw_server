/**
 * Common test helpers for E2E tests
 */
import { type Locator, type Page, type Route } from '@playwright/test';

/**
 * Environment configuration for tests
 */
export const TEST_CONFIG = {
  serverUrl: process.env.TLDW_SERVER_URL || 'http://127.0.0.1:8000',
  apiKey: process.env.TLDW_API_KEY || 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY',
  webUrl: process.env.TLDW_WEB_URL || 'http://localhost:8080',
  allowOffline: process.env.TLDW_E2E_ALLOW_OFFLINE !== '0',
};

const normalizeOrigin = (value: string): string => value.replace(/\/$/, '');

const resolveSeedServerUrl = (cfg: Partial<typeof TEST_CONFIG>): string => {
  if (typeof cfg.serverUrl === 'string' && cfg.serverUrl.trim().length > 0) {
    return normalizeOrigin(cfg.serverUrl.trim());
  }
  if (typeof cfg.webUrl === 'string' && cfg.webUrl.trim().length > 0) {
    return normalizeOrigin(cfg.webUrl.trim());
  }
  return normalizeOrigin(TEST_CONFIG.serverUrl);
};

const fulfillJson = async (route: Route, status: number, data: unknown): Promise<void> => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(data),
  });
};

/**
 * Seed authentication config in localStorage before page loads
 */
export async function seedAuth(
  page: Page,
  config: Partial<typeof TEST_CONFIG> = {}
): Promise<void> {
  const finalConfig = {
    ...TEST_CONFIG,
    ...config,
    serverUrl: resolveSeedServerUrl(config),
  };
  await page.addInitScript((cfg) => {
    const readStorageValue = (key: string) => {
      try {
        const raw = localStorage.getItem(key);
        if (raw == null) return undefined;
        return JSON.parse(raw);
      } catch {
        return localStorage.getItem(key) ?? undefined;
      }
    };

    const writeStorageValue = (key: string, value: unknown) => {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch {}
    };

    const installChromeStorageShim = () => {
      const globalWindow = window as unknown as {
        chrome?: Record<string, unknown>;
        browser?: Record<string, unknown>;
      };

      const listeners = new Set<
        (changes: Record<string, { oldValue: unknown; newValue: unknown }>, area: string) => void
      >();

      const emitChanges = (
        changes: Record<string, { oldValue: unknown; newValue: unknown }>,
        area: string
      ) => {
        for (const listener of listeners) {
          try {
            listener(changes, area);
          } catch {}
        }
      };

      const areaApi = {
        get: async (
          keys?: string | string[] | Record<string, unknown> | null,
          callback?: (result: Record<string, unknown>) => void
        ) => {
          let result: Record<string, unknown>;
          if (keys == null) {
            const out: Record<string, unknown> = {};
            for (let idx = 0; idx < localStorage.length; idx += 1) {
              const key = localStorage.key(idx);
              if (!key) continue;
              out[key] = readStorageValue(key);
            }
            result = out;
          } else if (typeof keys === "string") {
            result = { [keys]: readStorageValue(keys) };
          } else if (Array.isArray(keys)) {
            result = keys.reduce<Record<string, unknown>>((acc, key) => {
              acc[key] = readStorageValue(key);
              return acc;
            }, {});
          } else {
            result = Object.entries(keys).reduce<Record<string, unknown>>(
              (acc, [key, fallback]) => {
                const current = readStorageValue(key);
                acc[key] = typeof current === "undefined" ? fallback : current;
                return acc;
              },
              {}
            );
          }
          if (typeof callback === "function") {
            try {
              callback(result);
            } catch {}
          }
          return result;
        },
        set: async (items: Record<string, unknown>, callback?: () => void) => {
          const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {};
          for (const [key, value] of Object.entries(items || {})) {
            const oldValue = readStorageValue(key);
            writeStorageValue(key, value);
            changes[key] = { oldValue, newValue: value };
          }
          if (Object.keys(changes).length > 0) {
            emitChanges(changes, "sync");
          }
          if (typeof callback === "function") {
            try {
              callback();
            } catch {}
          }
        },
        remove: async (keys: string | string[], callback?: () => void) => {
          const values = Array.isArray(keys) ? keys : [keys];
          const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {};
          for (const key of values) {
            const oldValue = readStorageValue(key);
            try {
              localStorage.removeItem(key);
            } catch {}
            changes[key] = { oldValue, newValue: undefined };
          }
          if (Object.keys(changes).length > 0) {
            emitChanges(changes, "sync");
          }
          if (typeof callback === "function") {
            try {
              callback();
            } catch {}
          }
        },
        clear: async (callback?: () => void) => {
          const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {};
          for (let idx = 0; idx < localStorage.length; idx += 1) {
            const key = localStorage.key(idx);
            if (!key) continue;
            changes[key] = { oldValue: readStorageValue(key), newValue: undefined };
          }
          try {
            localStorage.clear();
          } catch {}
          if (Object.keys(changes).length > 0) {
            emitChanges(changes, "sync");
          }
          if (typeof callback === "function") {
            try {
              callback();
            } catch {}
          }
        },
        getBytesInUse: async (_keys?: unknown, callback?: (bytes: number) => void) => {
          if (typeof callback === "function") {
            try {
              callback(0);
            } catch {}
          }
          return 0;
        }
      };

      if (!globalWindow.chrome) {
        globalWindow.chrome = {};
      }
      const chromeLike = globalWindow.chrome as Record<string, unknown>;
      if (!chromeLike.runtime) {
        chromeLike.runtime = { id: "mock-runtime-id" };
      } else if (typeof (chromeLike.runtime as { id?: unknown }).id === "undefined") {
        (chromeLike.runtime as { id?: string }).id = "mock-runtime-id";
      }

      const storageShim = {
        sync: areaApi,
        local: areaApi,
        managed: areaApi,
        onChanged: {
          addListener: (
            fn: (changes: Record<string, { oldValue: unknown; newValue: unknown }>, area: string) => void
          ) => listeners.add(fn),
          removeListener: (
            fn: (changes: Record<string, { oldValue: unknown; newValue: unknown }>, area: string) => void
          ) => listeners.delete(fn)
        }
      };

      chromeLike.storage = storageShim;

      if (!globalWindow.browser) {
        globalWindow.browser = {};
      }
      const browserLike = globalWindow.browser as Record<string, unknown>;
      browserLike.storage = storageShim as Record<string, unknown>;
    };

    installChromeStorageShim();

    const authConfig = {
      serverUrl: cfg.serverUrl,
      authMode: 'single-user',
      apiKey: cfg.apiKey,
    };

    try {
      localStorage.setItem(
        'tldwConfig',
        JSON.stringify(authConfig)
      );
    } catch {}
    try {
      const chromeLike = (window as unknown as { chrome?: any }).chrome;
      chromeLike?.storage?.sync?.set?.({ tldwConfig: authConfig });
      chromeLike?.storage?.local?.set?.({ tldwConfig: authConfig });
      chromeLike?.storage?.sync?.set?.({ isMigrated: true });
      chromeLike?.storage?.local?.set?.({ isMigrated: true });
    } catch {}
    try {
      localStorage.setItem("isMigrated", "true");
    } catch {}
    try {
      localStorage.setItem('__tldw_first_run_complete', 'true');
    } catch {}
    try {
      if (cfg.allowOffline) {
        localStorage.setItem('__tldw_allow_offline', 'true');
      }
    } catch {}
    try {
      localStorage.setItem("serverUrl", cfg.serverUrl);
      localStorage.setItem("tldwServerUrl", cfg.serverUrl);
      localStorage.setItem("tldw-api-host", cfg.serverUrl);
      localStorage.setItem("authMode", "single-user");
      localStorage.setItem("apiKey", cfg.apiKey);
    } catch {}
    // Suppress connection error modals by setting test bypass
    try {
      localStorage.setItem('__tldw_test_bypass', 'true');
    } catch {}
  }, finalConfig);

  // Stub backend endpoints that may return 500 and trigger blocking error modals
  await page.route('**/api/v1/admin/notes/title-settings', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ llm_enabled: false, default_strategy: 'heuristic' }),
    });
  });
}

/**
 * Stub the notifications API family for suites that do not exercise
 * notifications behavior directly but still mount the global bridge.
 */
export async function stubNotificationsApi(page: Page): Promise<void> {
  await page.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const { pathname } = url;
    const method = request.method().toUpperCase();

    if (method === 'GET' && pathname === '/api/v1/notifications') {
      await fulfillJson(route, 200, {
        items: [],
        total: 0,
      });
      return;
    }

    if (method === 'GET' && pathname === '/api/v1/notifications/unread-count') {
      await fulfillJson(route, 200, {
        unread_count: 0,
      });
      return;
    }

    if (method === 'GET' && pathname === '/api/v1/notifications/preferences') {
      await fulfillJson(route, 200, {
        user_id: 'e2e-user',
        reminder_enabled: false,
        job_completed_enabled: false,
        job_failed_enabled: false,
        updated_at: new Date().toISOString(),
      });
      return;
    }

    if (method === 'GET' && pathname === '/api/v1/notifications/stream') {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: '',
      });
      return;
    }

    await fulfillJson(route, 200, {});
  });
}

/**
 * Wait for the app connection to be established
 */
export async function waitForConnection(page: Page, timeoutMs = 20000): Promise<void> {
  await page.waitForLoadState('domcontentloaded');

  // Wait for React root to mount
  const root = page.locator('#root, #__next');
  try {
    await root.waitFor({ state: 'attached', timeout: 15000 });
  } catch {
    // Root may take longer, connection check will timeout if app never mounts
  }

  // Wait for connection state
  try {
    await page.waitForFunction(
      () => {
        const store = (window as any).__tldw_useConnectionStore;
        const state = store?.getState?.().state;
        return state?.isConnected === true && state?.phase === 'connected';
      },
      undefined,
      { timeout: timeoutMs }
    );
  } catch {
    // Log connection snapshot for debugging
    await logConnectionState(page, 'connection-timeout');
  }

  // Dismiss any connection error modals that might block interaction
  await dismissConnectionModals(page);

}

/**
 * Wait for the app shell to mount enough DOM to interact with the route.
 * This is more stable than Playwright's networkidle on apps with polling/HMR.
 */
export async function waitForAppShell(page: Page, timeoutMs = 15000): Promise<void> {
  await page.waitForLoadState('domcontentloaded');

  const root = page.locator('#root, #__next');
  await root.first().waitFor({ state: 'attached', timeout: Math.min(timeoutMs, 15_000) }).catch(
    () => {}
  );

  await page
    .waitForFunction(
      () => {
        if (
          document.querySelector(
            [
              'main',
              '[role="main"]',
              '[data-testid="error-boundary"]',
              '[data-testid="not-found-recovery-panel"]',
              '[data-testid^="route-error-boundary-"]',
            ].join(', ')
          )
        ) {
          return true;
        }
        return (document.body?.innerText ?? '').trim().length > 0;
      },
      undefined,
      { timeout: timeoutMs }
    )
    .catch(() => {});
}

/**
 * Wait for the app shell plus a couple of paint cycles so screenshots and
 * post-navigation assertions don't depend on fixed sleeps.
 */
export async function waitForVisualSettle(page: Page, timeoutMs = 5000): Promise<void> {
  await waitForAppShell(page, timeoutMs);

  await page
    .evaluate(async () => {
      const docWithFonts = document as Document & {
        fonts?: {
          ready?: Promise<unknown>;
        };
      };

      try {
        await docWithFonts.fonts?.ready;
      } catch {}

      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => resolve());
        });
      });
    })
    .catch(() => {});
}

export function getAntdSelectTrigger(
  page: Page,
  options: {
    ariaLabel: string | RegExp;
  }
): Locator {
  // The ARIA combobox is the stable interactive target across Ant Design
  // render variants; the wrapper classes can disappear in some E2E builds.
  return page.getByRole('combobox', { name: options.ariaLabel }).first();
}
export function getVisibleAntdSelectOption(
  page: Page,
  options: {
    text: string | RegExp;
  }
): Locator {
  return page
    .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
    .filter({ hasText: options.text })
    .first();
}

export async function dispatchKeyboardShortcut(
  page: Page,
  options: {
    key: string;
    ctrlKey?: boolean;
    altKey?: boolean;
    shiftKey?: boolean;
    metaKey?: boolean;
  }
): Promise<void> {
  await page.evaluate((shortcut) => {
    const eventInit: KeyboardEventInit = {
      key: shortcut.key,
      bubbles: true,
      cancelable: true,
      ctrlKey: Boolean(shortcut.ctrlKey),
      altKey: Boolean(shortcut.altKey),
      shiftKey: Boolean(shortcut.shiftKey),
      metaKey: Boolean(shortcut.metaKey)
    }

    window.dispatchEvent(new KeyboardEvent("keydown", eventInit))
    document.dispatchEvent(new KeyboardEvent("keydown", eventInit))
  }, options)
}

/**
 * Dismiss any connection/server error modals (Ant Design modals).
 * Also removes the modal backdrop via DOM manipulation to prevent
 * modals from re-blocking interaction.
 */
export async function dismissConnectionModals(page: Page): Promise<void> {
  // Try clicking Dismiss button first
  try {
    const dismissBtn = page.getByRole('button', { name: /dismiss/i });
    if (await dismissBtn.isVisible({ timeout: 2_000 }).catch(() => false)) {
      await dismissBtn.click();
      await dismissBtn.waitFor({ state: 'hidden', timeout: 2_000 }).catch(() => {});
    }
  } catch {
    // No modal to dismiss
  }

  // Force-remove any remaining modal backdrops via DOM
  await page.evaluate(() => {
    document.querySelectorAll('.ant-modal-root, .ant-modal-wrap, .ant-modal-mask').forEach(el => {
      el.remove();
    });
    // Remove nextjs-portal if it has blocking overlays
    document.querySelectorAll('nextjs-portal').forEach(el => {
      if (el.children.length > 0) el.remove();
    });
    // Remove tldw portal root overlays
    const portalRoot = document.getElementById('tldw-portal-root');
    if (portalRoot) {
      portalRoot.querySelectorAll('.ant-modal-root, .ant-modal-wrap').forEach(el => el.remove());
    }
  }).catch(() => {});
}

/**
 * Log connection state for debugging
 */
export async function logConnectionState(page: Page, label: string): Promise<void> {
  await page.evaluate((tag) => {
    const store = (window as any).__tldw_useConnectionStore;
    if (!store?.getState) {
      console.log('CONNECTION_DEBUG', tag, JSON.stringify({ storeReady: false }));
      return;
    }
    try {
      const state = store.getState().state;
      console.log(
        'CONNECTION_DEBUG',
        tag,
        JSON.stringify({
          phase: state.phase,
          isConnected: state.isConnected,
          isChecking: state.isChecking,
          errorKind: state.errorKind,
        })
      );
    } catch {}
  }, label);
}

/**
 * Make authenticated API request
 */
export async function fetchWithApiKey(
  url: string,
  apiKey: string = TEST_CONFIG.apiKey,
  init: RequestInit = {}
): Promise<Response> {
  const headers = {
    'x-api-key': apiKey,
    ...(init.headers || {}),
  };
  return fetch(url, { ...init, headers });
}

/**
 * Backward-compatible page readiness helper for page objects.
 * Historically this waited on `networkidle`, which is brittle on polling pages.
 */
export async function waitForNetworkIdle(page: Page, timeoutMs = 30000): Promise<void> {
  await waitForAppShell(page, timeoutMs);
}

/**
 * Normalize URL by removing trailing slash
 */
export function normalizeUrl(url: string): string {
  return url.replace(/\/$/, '');
}

/**
 * Wait for element with retry logic
 */
export async function waitForElement(
  page: Page,
  selector: string,
  options: { timeout?: number; state?: 'attached' | 'visible' | 'hidden' } = {}
): Promise<void> {
  const { timeout = 10000, state = 'visible' } = options;
  const element = page.locator(selector);
  await element.waitFor({ state, timeout });
}

/**
 * Dismiss any visible modals or dialogs
 */
export async function dismissModals(page: Page): Promise<void> {
  // Dismiss Ant Design modals
  const modalCloseButtons = page.locator('.ant-modal-close');
  const count = await modalCloseButtons.count();
  for (let i = 0; i < count; i++) {
    const button = modalCloseButtons.nth(i);
    if (await button.isVisible()) {
      await button.click();
    }
  }
}

/**
 * Take a screenshot with a descriptive name
 */
export async function takeDebugScreenshot(page: Page, name: string): Promise<void> {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  await page.screenshot({
    path: `test-results/debug-${name}-${timestamp}.png`,
    fullPage: true,
  });
}

/**
 * Clear test data created during tests
 */
export async function cleanupTestData(
  page: Page,
  options: { conversations?: boolean; media?: boolean } = {}
): Promise<void> {
  // Clear localStorage items created by tests
  await page.evaluate((opts) => {
    if (opts.conversations) {
      // Clear conversation-related storage
      const keys = Object.keys(localStorage).filter(
        (k) => k.startsWith('conv_') || k.startsWith('chat_')
      );
      keys.forEach((k) => localStorage.removeItem(k));
    }
    if (opts.media) {
      // Clear media-related storage
      const keys = Object.keys(localStorage).filter(
        (k) => k.startsWith('media_') || k.startsWith('ingest_')
      );
      keys.forEach((k) => localStorage.removeItem(k));
    }
  }, options);
}

/**
 * Generate unique test identifiers
 */
export function generateTestId(prefix = 'test'): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `${prefix}-${timestamp}-${random}`;
}

/**
 * Patterns for console/error messages that are benign and should be ignored
 */
export const BENIGN_PATTERNS = [
  /ResizeObserver loop/,
  /Non-Error promise rejection/,
  /net::ERR_ABORTED/,
  /chrome-extension/,
  /Download the React DevTools/,
  /Fast Refresh/,
  /\[HMR\]/,
  /favicon\.ico.*404/,
  /Failed to load source map/,
  /Warning.*findDOMNode is deprecated/,
  /Hydration failed/,
  /There was an error while hydrating/,
  /cannot connect to an AudioNode belonging to a different audio context/i,
];

/**
 * Check if an error/warning message is benign
 */
export function isBenign(text: string): boolean {
  return BENIGN_PATTERNS.some((p) => p.test(text));
}

/**
 * Retry an action with exponential backoff
 */
export async function retry<T>(
  action: () => Promise<T>,
  options: { maxAttempts?: number; delayMs?: number } = {}
): Promise<T> {
  const { maxAttempts = 3, delayMs = 500 } = options;
  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await action();
    } catch (error) {
      lastError = error as Error;
      if (attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * Math.pow(2, attempt - 1)));
      }
    }
  }

  throw lastError;
}
