import type { Page } from '@playwright/test';
import { test, expect, assertNoCriticalErrors } from '../utils/fixtures';
import { expectNoHorizontalOverflow, seedAuth } from '../utils/helpers';

type ViewportTarget = {
  label: 'desktop' | 'mobile';
  width: number;
  height: number;
};

const VIEWPORTS: ViewportTarget[] = [
  { label: 'desktop', width: 1440, height: 900 },
  { label: 'mobile', width: 390, height: 844 },
];

const PLACEHOLDER_ROUTES = [
  {
    path: '/account',
    heading: /Hosted Account Pages Live In The Private Distribution/i,
  },
  {
    path: '/signup',
    heading: /Signup Is Not Part Of The OSS Web Surface/i,
  },
  {
    path: '/billing',
    heading: /Hosted Billing Lives In The Private Distribution/i,
  },
];

const hostedMode =
  String(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || '').trim().toLowerCase() ===
  'hosted';

async function seedFirstRunIncomplete(page: Page): Promise<void> {
  await seedAuth(page);
  await page.addInitScript(() => {
    localStorage.removeItem('__tldw_first_run_complete');
  });
}

async function expectOnePageHeading(page: Page, label: string): Promise<void> {
  const headings = page.getByRole('heading', { level: 1 });
  await expect(headings.first(), `${label} should expose a page heading`).toBeVisible({
    timeout: 15_000,
  });
  await expect(headings, `${label} should expose exactly one h1`).toHaveCount(1);
}

test.describe('Setup and recovery route QA', () => {
  for (const viewport of VIEWPORTS) {
    test(`first-run setup routes stay focused and fit the viewport (${viewport.label})`, async ({
      page,
      diagnostics,
    }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await seedFirstRunIncomplete(page);

      await page.goto('/', { waitUntil: 'domcontentloaded' });
      await expect(
        page.getByRole('heading', {
          level: 1,
          name: hostedMode
            ? /Start with the narrow hosted path, keep self-host when you need full control\./i
            : /Home Onboarding/i,
        })
      ).toBeVisible({
        timeout: 15_000,
      });
      await expectOnePageHeading(page, `${viewport.label} /`);
      await expectNoHorizontalOverflow(page, `${viewport.label} / first-run`);

      await page.goto('/setup', { waitUntil: 'domcontentloaded' });
      await expect(
        page.getByRole('heading', { level: 1, name: /Setup (Wizard|readiness)/i })
      ).toBeVisible({
        timeout: 15_000,
      });
      await expectOnePageHeading(page, `${viewport.label} /setup`);
      await expect(page.getByTestId('chat-header-theme-toggle')).toHaveCount(0);
      await expect(page.getByRole('navigation')).toHaveCount(0);
      await expectNoHorizontalOverflow(page, `${viewport.label} /setup`);

      await assertNoCriticalErrors(diagnostics);
    });

    test(`setup-adjacent recovery routes expose route context and fit the viewport (${viewport.label})`, async ({
      page,
      diagnostics,
    }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await seedAuth(page);

      await page.goto('/login?next=%2Faccount', { waitUntil: 'domcontentloaded' });
      if (hostedMode) {
        await expect(page).toHaveURL(/\/login\?next=%2Faccount/, {
          timeout: 15_000,
        });
      } else {
        await expect(page).toHaveURL(/\/settings\/tldw\?next=%2Faccount/, {
          timeout: 15_000,
        });
      }
      await expectNoHorizontalOverflow(page, `${viewport.label} /login redirect`);

      const expectedPrimaryText = hostedMode ? 'Open Login' : 'Open Local Auth Settings';
      const expectedPrimaryHref = hostedMode ? '/login' : '/settings/tldw';

      for (const route of PLACEHOLDER_ROUTES) {
        await page.goto(route.path, { waitUntil: 'domcontentloaded' });
        await expect(page.getByTestId('route-placeholder-panel')).toBeVisible({
          timeout: 15_000,
        });
        await expect(page.getByRole('heading', { level: 1, name: route.heading })).toBeVisible();
        await expectOnePageHeading(page, `${viewport.label} ${route.path}`);
        await expect(page.getByText(route.path, { exact: true })).toHaveCount(2);
        await expect(page.getByTestId('route-placeholder-primary')).toHaveText(expectedPrimaryText);
        await expect(page.getByTestId('route-placeholder-primary')).toHaveAttribute(
          'href',
          expectedPrimaryHref
        );
        await expectNoHorizontalOverflow(page, `${viewport.label} ${route.path}`);
      }

      const missingRoute = `/missing-setup-route-${viewport.label}`;
      await page.goto(missingRoute, { waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('not-found-recovery-panel')).toBeVisible({
        timeout: 15_000,
      });
      await expect(
        page.getByRole('heading', { level: 1, name: 'We could not find that route' })
      ).toBeVisible();
      await expectOnePageHeading(page, `${viewport.label} ${missingRoute}`);
      await expect(page.getByText(missingRoute)).toBeVisible();
      await expect(page.getByTestId('not-found-open-home')).toHaveText('Open Home');
      await expectNoHorizontalOverflow(page, `${viewport.label} ${missingRoute}`);

      await assertNoCriticalErrors(diagnostics);
    });
  }
});
