import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from './smoke.setup';
import type { Page } from '@playwright/test';
import { waitForAppShell, waitForVisualSettle } from '../utils/helpers';
import { PAGES } from './page-inventory';
import {
  getRouteHeadingPolicy,
  getRouteMetadata,
} from '../../../packages/ui/src/routes/route-metadata';

const PRIMARY_HEADING_ROUTES = PAGES
  .filter((entry) => !entry.skip)
  .map((entry) => ({
    path: entry.path,
    metadata: getRouteMetadata(entry.path),
  }))
  .filter(({ metadata }) => metadata?.smoke === 'include')
  .filter(({ metadata }) => metadata?.surface === 'default_self_hosted')
  .filter(({ metadata }) => metadata?.nav === 'primary')
  .filter(({ metadata }) => metadata && getRouteHeadingPolicy(metadata).requiresH1)
  .map(({ path, metadata }) => ({
    path,
    label: metadata?.label ?? path,
  }));

async function installRouteHeadingMocks(page: Page): Promise<void> {
  await page.route('**/api/v1/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'healthy', checks: {} }),
    });
  });

  await page.route('**/api/v1/health/live', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok' }),
    });
  });

  await page.route('**/api/v1/media**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname !== '/api/v1/media' && url.pathname !== '/api/v1/media/') {
      await route.fallback();
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [],
        pagination: {
          page: 1,
          results_per_page: 20,
          total_items: 0,
          total_pages: 1,
        },
      }),
    });
  });
}

test.describe('route heading governance', () => {
  test('has at least one primary route requiring h1 governance', () => {
    expect(PRIMARY_HEADING_ROUTES.length).toBeGreaterThan(0);
  });

  for (const entry of PRIMARY_HEADING_ROUTES) {
    test(`${entry.path} exposes one semantic h1`, async ({ page }) => {
      await installRouteHeadingMocks(page);
      await seedAuth(page);
      await page.goto(entry.path, {
        waitUntil: 'domcontentloaded',
        timeout: SMOKE_LOAD_TIMEOUT,
      });
      await waitForAppShell(page, SMOKE_LOAD_TIMEOUT);
      await waitForVisualSettle(page, SMOKE_LOAD_TIMEOUT);

      const headings = page.locator('h1');
      await expect(
        headings,
        `${entry.path} (${entry.label}) must expose exactly one semantic h1`
      ).toHaveCount(1);
    });
  }
});
