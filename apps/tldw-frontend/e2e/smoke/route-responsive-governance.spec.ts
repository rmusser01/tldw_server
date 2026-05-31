import type { Page, Route } from '@playwright/test';
import {
  test,
  expect,
  seedAuth,
  getCriticalIssues,
  SMOKE_LOAD_TIMEOUT,
  type DiagnosticsData,
} from './smoke.setup';
import { stubNotificationsApi, waitForAppShell, waitForVisualSettle } from '../utils/helpers';
import { PAGES } from './page-inventory';
import {
  getRouteMetadata,
  normalizeRoutePath,
  type RouteAvailability,
  type RouteMetadata,
} from '../../../packages/ui/src/routes/route-metadata';

const RESPONSIVE_ROUTE_PATHS = [
  '/chat',
  '/media',
  '/settings',
  '/settings/model',
  '/prompts',
  '/research-workspace',
  '/setup',
  '/sources',
  '/mcp-hub',
  '/stt',
  '/tts',
  '/chat-workspace',
] as const;

const SIDEPANEL_ROUTE_PATHS = ['/chat', '/flashcards', '/companion', '/persona'] as const;

const PAGE_LEVEL_VIEWPORT = { width: 390, height: 844 };
const SIDEPANEL_VIEWPORT = { width: 360, height: 720 };
// Matches the Stage 4 responsive suite tolerance for sub-pixel and scrollbar variance.
const MAX_HORIZONTAL_OVERFLOW_PX = 4;

type GovernanceRoute = {
  path: string;
  label: string;
};

const normalizedInventory = new Map(PAGES.map((page) => [normalizeRoutePath(page.path), page]));

const responsiveRoutes: GovernanceRoute[] = RESPONSIVE_ROUTE_PATHS.map((path) => ({
  path,
  label: getRouteMetadata(path)?.label ?? path,
}));

const sidepanelRoutes: GovernanceRoute[] = SIDEPANEL_ROUTE_PATHS.map((path) => ({
  path,
  label: getRouteMetadata(path)?.label ?? path,
}));

const json = async (route: Route, status: number, body: unknown): Promise<void> => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });
};

const getMetadataProblem = (
  path: string,
  metadata: RouteMetadata | undefined,
  requiredAvailability: RouteAvailability
): string | undefined => {
  const inventoryEntry = normalizedInventory.get(normalizeRoutePath(path));

  if (!inventoryEntry) {
    return `${path} is missing from page-inventory.ts`;
  }

  if (inventoryEntry.skip) {
    return `${path} is skipped in page-inventory.ts: ${inventoryEntry.skip}`;
  }

  if (!metadata) {
    return `${path} is missing route metadata`;
  }

  if (!metadata.availability.includes(requiredAvailability)) {
    return `${path} metadata does not include ${requiredAvailability} availability`;
  }

  if (metadata.smoke === 'exclude') {
    return `${path} metadata is excluded from smoke coverage`;
  }

  if (
    metadata.surface === 'legacy_alias' ||
    metadata.surface === 'redirect' ||
    metadata.surface === 'deprecated'
  ) {
    return `${path} metadata delegates ownership to ${metadata.surface}`;
  }

  return undefined;
};

const getResponsiveMetadataProblems = (): string[] =>
  RESPONSIVE_ROUTE_PATHS.flatMap((path) => {
    const problem = getMetadataProblem(path, getRouteMetadata(path), 'web');
    return problem ? [problem] : [];
  });

const getSidepanelMetadataProblems = (): string[] =>
  SIDEPANEL_ROUTE_PATHS.flatMap((path) => {
    const problem = getMetadataProblem(path, getRouteMetadata(path), 'extension_sidepanel');
    return problem ? [problem] : [];
  });

const emptyPaginatedResponse = () => ({
  items: [],
  pagination: {
    page: 1,
    results_per_page: 20,
    total_items: 0,
    total_pages: 1,
  },
});

async function installResponsiveGovernanceBackend(page: Page): Promise<void> {
  await page.addInitScript(() => {
    localStorage.removeItem('__tldwServerCapabilitiesCacheV5');
    sessionStorage.removeItem('__tldwServerCapabilitiesCacheV5');
  });

  await stubNotificationsApi(page);

  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const { pathname } = url;

    if (pathname === '/openapi.json') {
      await json(route, 200, {
        openapi: '3.0.0',
        info: { title: 'tldw responsive governance fixture', version: 'e2e' },
        paths: {},
      });
      return;
    }

    if (pathname === '/api/v1/health' || pathname === '/api/v1/health/live') {
      await json(route, 200, { status: 'ok', checks: {} });
      return;
    }

    if (pathname === '/api/v1/config/docs-info') {
      await json(route, 200, {
        info: { version: 'e2e' },
        capabilities: {
          hasAudio: true,
          hasStt: true,
          hasTts: true,
        },
      });
      return;
    }

    if (pathname === '/api/v1/users/keys') {
      await json(route, 200, { keys: [] });
      return;
    }

    if (pathname === '/api/v1/users/keys/openai/oauth/status') {
      await json(route, 200, { connected: false });
      return;
    }

    if (pathname === '/api/v1/llm/providers') {
      await json(route, 200, { providers: [] });
      return;
    }

    if (pathname === '/api/v1/llm/models/metadata') {
      await json(route, 200, { models: [], total: 0 });
      return;
    }

    if (pathname === '/api/v1/media' || pathname === '/api/v1/media/') {
      await json(route, 200, emptyPaginatedResponse());
      return;
    }

    if (pathname === '/api/v1/prompts') {
      await json(route, 200, { prompts: [], total: 0 });
      return;
    }

    if (pathname === '/api/v1/sources') {
      await json(route, 200, { sources: [], total: 0 });
      return;
    }

    if (pathname === '/api/v1/mcp/hub/tool-registry/summary') {
      await json(route, 200, { entries: [], modules: [] });
      return;
    }

    if (pathname === '/api/v1/mcp/hub/external-servers') {
      await json(route, 200, []);
      return;
    }

    if (pathname === '/api/v1/audio/providers') {
      await json(route, 200, { providers: {}, voices: {} });
      return;
    }

    if (pathname === '/api/v1/audio/voices/catalog') {
      await json(route, 200, { voices: [] });
      return;
    }

    if (pathname.startsWith('/api/v1/')) {
      await json(route, 200, {});
      return;
    }

    await route.continue();
  });
}

async function visitGovernedRoute(
  page: Page,
  path: string,
  viewport: { width: number; height: number }
): Promise<void> {
  await page.setViewportSize(viewport);
  await installResponsiveGovernanceBackend(page);
  await seedAuth(page);

  await page.goto(path, {
    waitUntil: 'domcontentloaded',
    timeout: SMOKE_LOAD_TIMEOUT,
  });
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT);
  await waitForVisualSettle(page, SMOKE_LOAD_TIMEOUT);
}

async function expectGovernedRouteRendered(
  page: Page,
  diagnostics: DiagnosticsData,
  path: string
): Promise<void> {
  await expect(
    page.locator('[data-testid="error-boundary"], [data-testid^="route-error-boundary-"]'),
    `${path} rendered an error boundary instead of governed route content`
  ).toHaveCount(0);

  const issues = getCriticalIssues(diagnostics);
  expect(issues.pageErrors, `Uncaught page errors while rendering ${path}`).toHaveLength(0);
}

async function getPageLevelHorizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(() => {
    const documentElement = document.documentElement;
    const body = document.body;
    const scrollingElement = document.scrollingElement ?? documentElement;
    const viewportWidth = documentElement.clientWidth;
    const scrollWidth = Math.max(
      scrollingElement.scrollWidth,
      documentElement.scrollWidth,
      body?.scrollWidth ?? 0
    );

    return Math.max(0, scrollWidth - viewportWidth);
  });
}

test.describe('responsive route governance', () => {
  test('planned 390px routes are active web route metadata entries', () => {
    expect(getResponsiveMetadataProblems()).toEqual([]);
  });

  test('sidepanel-width routes are backed by extension sidepanel metadata', () => {
    expect(getSidepanelMetadataProblems()).toEqual([]);
  });

  for (const route of responsiveRoutes) {
    test(`${route.path} (${route.label}) has no page-level overflow at 390px`, async ({
      page,
      diagnostics,
    }) => {
      await visitGovernedRoute(page, route.path, PAGE_LEVEL_VIEWPORT);
      await expectGovernedRouteRendered(page, diagnostics, route.path);

      const horizontalOverflow = await getPageLevelHorizontalOverflow(page);
      expect(
        horizontalOverflow,
        `${route.path} introduced ${horizontalOverflow}px of horizontal overflow at 390px`
      ).toBeLessThanOrEqual(MAX_HORIZONTAL_OVERFLOW_PX);
    });
  }

  for (const route of sidepanelRoutes) {
    test(`${route.path} (${route.label}) has no sidepanel-width overflow`, async ({
      page,
      diagnostics,
    }) => {
      await visitGovernedRoute(page, route.path, SIDEPANEL_VIEWPORT);
      await expectGovernedRouteRendered(page, diagnostics, route.path);

      const horizontalOverflow = await getPageLevelHorizontalOverflow(page);
      expect(
        horizontalOverflow,
        `${route.path} introduced ${horizontalOverflow}px of horizontal overflow at sidepanel width`
      ).toBeLessThanOrEqual(MAX_HORIZONTAL_OVERFLOW_PX);
    });
  }
});
