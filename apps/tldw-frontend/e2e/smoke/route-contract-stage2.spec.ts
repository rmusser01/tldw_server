import fs from 'node:fs/promises';
import path from 'node:path';
import { test, expect, seedAuth } from './smoke.setup';
import { waitForAppShell } from '../utils/helpers';
import {
  getRouteMetadata,
  getRoutesForSmokeInventory,
  ROUTE_METADATA
} from '../../../packages/ui/src/routes/route-metadata';
import { PAGES } from './page-inventory';

const LOAD_TIMEOUT = 30_000;
const OUTPUT_DATE = process.env.TLDW_STAGE2_OUTPUT_DATE;
const OUTPUT_SUFFIX = process.env.TLDW_STAGE2_OUTPUT_SUFFIX;
const OUTPUT_FILE = OUTPUT_DATE
  ? `stage2_route_contract_check_${OUTPUT_DATE}${OUTPUT_SUFFIX ? `_${OUTPUT_SUFFIX}` : ''}.json`
  : 'stage2_route_contract_check_2026-02-16.json';
const OUTPUT_PATH = path.resolve(process.cwd(), `../../Docs/Plans/artifacts/${OUTPUT_FILE}`);

type RouteContract = {
  route: string;
  expectedTitle: string;
  disallowedFinalPaths: string[];
  expectedUi: 'placeholder' | 'real';
};

type RouteContractResult = {
  route: string;
  status: number;
  finalPath: string;
  redirected: boolean;
  hasPlaceholderPanel: boolean;
  hasRedirectPanel: boolean;
  hasExpectedTitle: boolean;
  hasAdminGuard: boolean;
};

const ROUTE_CONTRACTS: RouteContract[] = [
  {
    route: '/admin/data-ops',
    expectedTitle: 'Data Operations',
    disallowedFinalPaths: ['/admin/server'],
    expectedUi: 'real',
  },
  {
    route: '/admin/watchlists-runs',
    expectedTitle: 'Watchlist Runs Admin Is Coming Soon',
    disallowedFinalPaths: ['/admin/server'],
    expectedUi: 'placeholder',
  },
  {
    route: '/admin/watchlists-items',
    expectedTitle: 'Watchlists Items',
    disallowedFinalPaths: ['/admin/server'],
    expectedUi: 'real',
  },
  {
    route: '/connectors',
    expectedTitle: 'Connectors',
    disallowedFinalPaths: ['/settings'],
    expectedUi: 'placeholder',
  },
  {
    route: '/connectors/sources',
    expectedTitle: 'Connector Sources',
    disallowedFinalPaths: ['/settings'],
    expectedUi: 'placeholder',
  },
  {
    route: '/connectors/jobs',
    expectedTitle: 'Connector Jobs',
    disallowedFinalPaths: ['/settings'],
    expectedUi: 'placeholder',
  },
  {
    route: '/connectors/browse',
    expectedTitle: 'Connector Catalog',
    disallowedFinalPaths: ['/settings'],
    expectedUi: 'placeholder',
  },
  {
    route: '/profile',
    expectedTitle: 'Profile Page Is Coming Soon',
    disallowedFinalPaths: ['/settings'],
    expectedUi: 'placeholder',
  },
  {
    route: '/config',
    expectedTitle: 'Configuration Center Is Coming Soon',
    disallowedFinalPaths: ['/settings'],
    expectedUi: 'placeholder',
  },
  {
    route: '/admin/orgs',
    expectedTitle: 'Organizations & Teams',
    disallowedFinalPaths: ['/admin/server'],
    expectedUi: 'real',
  },
  {
    route: '/admin/maintenance',
    expectedTitle: 'Maintenance Console',
    disallowedFinalPaths: ['/admin/server'],
    expectedUi: 'real',
  },
];

test.describe('Route metadata smoke inventory contract', () => {
  const pageEntryByPath = new Map(PAGES.map((entry) => [entry.path, entry]));

  test('keeps included smoke routes in the page inventory', () => {
    const missingIncludedRoutes = ROUTE_METADATA
      .filter((metadata) => metadata.availability.includes('web'))
      .filter((metadata) => metadata.smoke === 'include')
      .filter((metadata) => !pageEntryByPath.has(metadata.path))
      .map((metadata) => metadata.path);

    expect(missingIncludedRoutes).toEqual([]);
  });

  test('does not treat internal QA/debug metadata as normal product pages', () => {
    const activeDebugPages = ROUTE_METADATA
      .filter((metadata) => metadata.surface === 'internal_qa_debug')
      .filter((metadata) => {
        const pageEntry = pageEntryByPath.get(metadata.path);

        return pageEntry && !pageEntry.skip;
      })
      .map((metadata) => metadata.path);

    expect(activeDebugPages).toEqual([]);
  });

  test('keeps inventory aliases tied to canonical metadata rows', () => {
    const aliasesWithoutCanonicalMetadata = PAGES
      .map((entry) => ({
        entry,
        metadata: getRouteMetadata(entry.path),
      }))
      .filter(({ metadata }) => metadata && metadata.canonicalPath !== metadata.path)
      .filter(({ metadata }) => !getRouteMetadata(metadata?.canonicalPath ?? ''))
      .map(({ entry }) => entry.path);

    expect(aliasesWithoutCanonicalMetadata).toEqual([]);
  });
});

test.describe('Stage 2 route contracts', () => {
  test('route metadata smoke policy is represented in the page inventory', async () => {
    const pagePaths = new Set(PAGES.map((pageEntry) => pageEntry.path));
    const duplicatePagePaths = PAGES.map((pageEntry) => pageEntry.path).filter(
      (routePath, index, paths) => paths.indexOf(routePath) !== index
    );
    const missingSmokeRoutes = getRoutesForSmokeInventory().filter(
      (routePath) => !pagePaths.has(routePath)
    );
    const manualRoutesWithoutReason = ROUTE_METADATA.filter(
      (metadata) => metadata.smoke === 'manual' && !metadata.rationale.trim()
    ).map((metadata) => metadata.path);
    const internalRoutesInSmoke = ROUTE_METADATA.filter(
      (metadata) =>
        metadata.surface === 'internal_qa_debug' && metadata.smoke !== 'exclude'
    ).map((metadata) => metadata.path);
    const excludedRoutesInInventory = ROUTE_METADATA.filter(
      (metadata) => metadata.smoke === 'exclude' && pagePaths.has(metadata.path)
    ).map((metadata) => metadata.path);

    expect(duplicatePagePaths).toEqual([]);
    expect(missingSmokeRoutes).toEqual([]);
    expect(manualRoutesWithoutReason).toEqual([]);
    expect(internalRoutesInSmoke).toEqual([]);
    expect(excludedRoutesInInventory).toEqual([]);
  });

  test('routes render their intended surface and do not misroute', async ({ page }, testInfo) => {
    await seedAuth(page);
    const results: RouteContractResult[] = [];

    for (const contract of ROUTE_CONTRACTS) {
      const response = await page.goto(contract.route, {
        waitUntil: 'domcontentloaded',
        timeout: LOAD_TIMEOUT,
      });
      await waitForAppShell(page, LOAD_TIMEOUT);

      const placeholderPanel = page.getByTestId('route-placeholder-panel');
      const redirectPanel = page.getByTestId('route-redirect-panel');
      const expectedHeading = page.getByRole('heading', { name: contract.expectedTitle });
      const adminGuardAlert = page.locator('.ant-alert-warning, .ant-alert-error').first();

      if (contract.expectedUi === 'placeholder') {
        await expect(
          placeholderPanel,
          `Expected ${contract.route} to mount its placeholder panel`
        ).toBeVisible({ timeout: LOAD_TIMEOUT });
      } else {
        await expect(
          expectedHeading.or(adminGuardAlert),
          `Expected ${contract.route} to mount either heading "${contract.expectedTitle}" or an admin guard alert`
        ).toBeVisible({ timeout: LOAD_TIMEOUT });
      }

      const status = response?.status() ?? 0;
      const finalPath = new URL(page.url()).pathname;
      const hasPlaceholderPanel = await placeholderPanel.isVisible().catch(() => false);
      const hasRedirectPanel = (await redirectPanel.count()) > 0;
      const hasExpectedTitle = await expectedHeading.isVisible().catch(() => false);
      const hasAdminGuard = await adminGuardAlert.isVisible().catch(() => false);
      const redirected = finalPath !== contract.route;

      results.push({
        route: contract.route,
        status,
        finalPath,
        redirected,
        hasPlaceholderPanel,
        hasRedirectPanel,
        hasExpectedTitle,
        hasAdminGuard,
      });

      expect(
        status,
        `Expected HTTP 2xx/3xx for ${contract.route}, received status ${status}`
      ).toBeGreaterThanOrEqual(200);
      expect(
        status,
        `Expected HTTP 2xx/3xx for ${contract.route}, received status ${status}`
      ).toBeLessThan(400);

      expect(
        finalPath,
        `Expected ${contract.route} to remain on its own route and not auto-redirect`
      ).toBe(contract.route);
      expect(
        contract.disallowedFinalPaths.includes(finalPath),
        `Route ${contract.route} unexpectedly landed on disallowed path ${finalPath}`
      ).toBe(false);
      expect(
        hasRedirectPanel,
        `Expected ${contract.route} to avoid RouteRedirect panel`
      ).toBe(false);

      if (contract.expectedUi === 'placeholder') {
        expect(
          hasPlaceholderPanel,
          `Expected ${contract.route} to render route placeholder panel`
        ).toBe(true);
        expect(
          hasExpectedTitle,
          `Expected ${contract.route} placeholder title to be "${contract.expectedTitle}"`
        ).toBe(true);
      } else {
        expect(
          hasPlaceholderPanel,
          `Expected ${contract.route} to avoid the route placeholder panel because it now renders a real page`
        ).toBe(false);
        expect(
          hasExpectedTitle || hasAdminGuard,
          `Expected ${contract.route} to render either heading "${contract.expectedTitle}" or an admin guard alert`
        ).toBe(true);
      }
    }

    const artifact = {
      generatedAt: new Date().toISOString(),
      baseUrl: testInfo.project.use.baseURL,
      routeCount: ROUTE_CONTRACTS.length,
      results,
    };

    await fs.writeFile(OUTPUT_PATH, `${JSON.stringify(artifact, null, 2)}\n`, 'utf8');
    testInfo.annotations.push({
      type: 'stage2-route-contract-artifact',
      description: OUTPUT_PATH,
    });
  });
});
