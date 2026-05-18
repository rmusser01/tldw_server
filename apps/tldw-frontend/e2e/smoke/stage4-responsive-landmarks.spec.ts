import type { Page } from '@playwright/test';
import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from './smoke.setup';
import { waitForAppShell, waitForVisualSettle } from '../utils/helpers';

type ResponsiveRouteMatrixEntry = {
  path: string;
  heading: RegExp;
  allowRedirect?: boolean;
};

const RESPONSIVE_ROUTE_MATRIX: ResponsiveRouteMatrixEntry[] = [
  { path: '/chat', heading: /^Chat$/i },
  { path: '/media', heading: /^Media Inspector$/i },
  { path: '/settings', heading: /^Settings$/i },
  { path: '/settings/model', heading: /^Model settings$/i },
  { path: '/prompts', heading: /^Prompts$/i },
  { path: '/workspace-playground', heading: /^New Research$/i },
  { path: '/setup', heading: /setup/i, allowRedirect: true },
  { path: '/sources', heading: /sources/i },
  { path: '/mcp-hub', heading: /mcp hub/i },
  { path: '/stt', heading: /speech to text/i },
  { path: '/tts', heading: /text to speech/i },
  { path: '/chat-workspace', heading: /chat workspace/i },
];

// Allows minor sub-pixel and scrollbar variance while still catching real page overflow.
const OVERFLOW_TOLERANCE_PX = 4;

type OverflowOffender = {
  testId: string | null;
  tag: string;
  className: string;
  scrollWidth: number;
  clientWidth: number;
  rectWidth: number;
};

async function installReadinessMocks(page: Page): Promise<void> {
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

async function seedResponsiveRouteState(page: Page): Promise<void> {
  await seedAuth(page);
  await page.addInitScript(() => {
    localStorage.setItem('stickyChatInput', 'true');
    localStorage.setItem('playgroundComposerOptionsExpanded', 'false');
  });
}

async function gotoResponsiveRoute(page: Page, entry: ResponsiveRouteMatrixEntry): Promise<string> {
  await installReadinessMocks(page);
  await seedResponsiveRouteState(page);
  await page.setViewportSize({ width: 390, height: 844 });

  await page.goto(entry.path, {
    waitUntil: 'domcontentloaded',
    timeout: SMOKE_LOAD_TIMEOUT,
  });
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT);
  await waitForVisualSettle(page, SMOKE_LOAD_TIMEOUT);

  const finalPath = new URL(page.url()).pathname;
  if (entry.allowRedirect && finalPath !== entry.path) {
    test.info().annotations.push({
      type: 'responsive-route-redirect',
      description: `${entry.path} redirected to ${finalPath}`,
    });
  }

  return finalPath;
}

async function expectOneRouteHeading(
  page: Page,
  entry: ResponsiveRouteMatrixEntry,
  finalPath: string
): Promise<void> {
  if (entry.allowRedirect && finalPath !== entry.path) return;

  const headings = page.locator('h1');
  await expect(headings, `${entry.path} must expose exactly one semantic h1`).toHaveCount(1);
  await expect(headings.first(), `${entry.path} h1 text`).toHaveText(entry.heading);
}

async function expectNoPageHorizontalOverflow(page: Page, routePath: string): Promise<void> {
  const overflowState = await page.evaluate((overflowTolerancePx) => {
    const root = document.documentElement;
    const rootScrollWidth = root.scrollWidth;
    const viewportWidth = window.innerWidth;
    const offenders =
      rootScrollWidth > viewportWidth + overflowTolerancePx
        ? Array.from(document.querySelectorAll<HTMLElement>('body *'))
            .map((element) => {
              const rect = element.getBoundingClientRect();
              return {
                testId: element.getAttribute('data-testid'),
                tag: element.tagName.toLowerCase(),
                className: String(element.className || ''),
                scrollWidth: element.scrollWidth,
                clientWidth: element.clientWidth,
                rectWidth: rect.width,
              };
            })
            .filter(
              (entry) =>
                entry.scrollWidth > viewportWidth + overflowTolerancePx ||
                entry.rectWidth > viewportWidth + overflowTolerancePx
            )
            .slice(0, 10)
        : [];

    return {
      rootScrollWidth,
      viewportWidth,
      offenders,
    };
  }, OVERFLOW_TOLERANCE_PX);

  expect(
    overflowState.rootScrollWidth,
    `${routePath} page overflowed viewport: ${JSON.stringify(
      overflowState.offenders satisfies OverflowOffender[],
      null,
      2
    )}`
  ).toBeLessThanOrEqual(overflowState.viewportWidth + OVERFLOW_TOLERANCE_PX);
}

async function expectChatComposerInsideViewport(page: Page): Promise<void> {
  const nextGenComposer = page.getByTestId('nextgen-composer-wrapper');
  const composer =
    (await nextGenComposer.count()) > 0
      ? nextGenComposer
      : page.getByTestId('playground-chat-composer-dock');
  await expect(composer).toBeVisible({ timeout: SMOKE_LOAD_TIMEOUT });

  const input = page.getByTestId('chat-input');
  await expect(input).toBeVisible({ timeout: SMOKE_LOAD_TIMEOUT });
  await input.focus();

  const box = await composer.boundingBox();
  expect(box, 'Expected /chat composer wrapper to have a layout box').not.toBeNull();
  expect(box?.y ?? 0).toBeGreaterThanOrEqual(0);
  expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(844);
}

test.describe('Stage 4 responsive route landmarks', () => {
  for (const entry of RESPONSIVE_ROUTE_MATRIX) {
    test(`${entry.path} has one route heading and no page overflow at 390px`, async ({ page }) => {
      const finalPath = await gotoResponsiveRoute(page, entry);

      await expectOneRouteHeading(page, entry, finalPath);
      await expectNoPageHorizontalOverflow(page, entry.path);

      if (entry.path === '/chat') {
        await expectChatComposerInsideViewport(page);
      }
    });
  }
});
