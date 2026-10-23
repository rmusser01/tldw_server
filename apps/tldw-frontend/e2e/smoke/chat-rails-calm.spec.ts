import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const prepareOfflinePreview = async (page: Page) => {
  await page.route('http://127.0.0.1:8000/**', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Offline preview' }),
    })
  );
  await page.route('**/api/v1/llm/models/metadata**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ models: [] }),
    })
  );
  await page.route('**/api/v1/llm/providers**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ providers: [] }),
    })
  );
  await page.addInitScript(() => {
    localStorage.setItem('assistant_setup_dismissed', 'true');
    localStorage.setItem('__tldw_first_run_complete', 'true');
    localStorage.setItem('__tldw_test_bypass', 'true');
    localStorage.setItem('stickyChatInput', 'true');
    localStorage.setItem(
      'tldwConfig',
      JSON.stringify({
        serverUrl: 'http://127.0.0.1:8000',
        authMode: 'single-user',
        apiKey: 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY',
      })
    );
  });
};

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  for (const theme of ['dark', 'light'] as const) {
    test(`calm rails ${viewport.name} ${theme}`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await prepareOfflinePreview(page);
      await page.addInitScript((value) => {
        localStorage.setItem('theme', value);
        localStorage.setItem('playgroundChatContextRailVisible', 'true');
        localStorage.setItem('playgroundChatRuntimeRailVisible', 'true');
        localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
      }, theme);
      await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
      await page.setViewportSize(viewport);
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      const input = page.getByTestId('chat-input');
      await expect(input).toBeVisible({ timeout: 60_000 });
      const exitFocus = page.getByRole('button', { name: 'Exit focus', exact: true });
      if (await exitFocus.isVisible()) await exitFocus.click();
      await expect(page.locator('html')).toHaveClass(theme === 'dark' ? /dark/ : /light/);
      await expect(page.getByTestId('model-selector')).toBeVisible();
      await expect(page.getByRole('button', { name: 'Queue request', exact: true })).toBeVisible();
      await page.evaluate(async () => {
        await document.fonts.ready;
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
      });
      const mobile = viewport.name === 'mobile';
      if (mobile) await page.getByRole('tab', { name: 'Context', exact: true }).click();
      const context = mobile
        ? page.getByRole('tabpanel', { name: 'Context', exact: true })
        : page.getByTestId('playground-cockpit-left-rail');
      await expect(
        context.getByRole('heading', { name: 'Instructions', exact: true })
      ).toBeVisible();
      await expect(context.getByRole('region', { name: 'Assistant', exact: true })).toBeVisible();
      await expect(
        context.getByRole('button', { name: 'Select character or persona', exact: true })
      ).toBeVisible();
      const preview = context.getByRole('button', { name: 'Expand Request preview', exact: true });
      await expect(preview).toHaveAttribute('aria-expanded', 'false');
      const contextAxe = await new AxeBuilder({ page })
        .include(
          mobile
            ? '[role="tabpanel"]:not([hidden])'
            : '[data-testid="playground-cockpit-left-rail"]'
        )
        .analyze();
      expect(contextAxe.violations).toEqual([]);
      if (mobile)
        await page.screenshot({
          animations: 'disabled',
          path: testInfo.outputPath(`${viewport.name}-${theme}-context.png`),
        });
      if (mobile) await page.getByRole('tab', { name: 'Runtime', exact: true }).click();
      const runtime = mobile
        ? page.getByRole('tabpanel', { name: 'Runtime', exact: true })
        : page.getByTestId('playground-cockpit-right-rail');
      await expect(
        runtime.getByRole('heading', { name: 'Current run', exact: true })
      ).toBeVisible();
      await expect(
        runtime.getByRole('button', { name: 'Stop generation', exact: true })
      ).toBeVisible();
      await expect(
        runtime.getByRole('button', { name: 'Regenerate last response', exact: true })
      ).toBeVisible();
      await expect(runtime.getByRole('region', { name: 'Assistant', exact: true })).toHaveCount(0);
      const runtimeAxe = await new AxeBuilder({ page })
        .include(
          mobile
            ? '[role="tabpanel"]:not([hidden])'
            : '[data-testid="playground-cockpit-right-rail"]'
        )
        .analyze();
      expect(runtimeAxe.violations).toEqual([]);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      expect(overflow).toBe(false);
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath(`${viewport.name}-${theme}-${mobile ? 'runtime' : 'rails'}.png`),
      });
      await input.fill('Keep this draft while I inspect the rails.');
      await runtime.getByRole('button', { name: 'Open model settings', exact: true }).click();
      const dialog = page.getByRole('dialog').last();
      await expect(dialog).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(dialog).toBeHidden();
      await expect(input).toHaveValue('Keep this draft while I inspect the rails.');
      if (mobile) {
        await page.getByRole('button', { name: 'Close Runtime panel', exact: true }).click();
        await expect(runtime).toBeHidden();
        await expect(page.getByRole('tab', { name: 'Runtime', exact: true })).toBeFocused();
        await page.getByRole('tab', { name: 'Context', exact: true }).click();
        await expect(context).toBeVisible();
        await expect(input).toHaveValue('Keep this draft while I inspect the rails.');
      }
      expect(errors).toEqual([]);
    });
  }
}

test('keeps independent rail visibility after reload', async ({ page }) => {
  await prepareOfflinePreview(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 60_000 });
  await page.getByTestId('playground-cockpit-left-rail-restore').click();
  await page.getByTestId('playground-cockpit-right-rail-restore').click();
  await page.getByRole('button', { name: 'Collapse context sidechannel', exact: true }).click();
  await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('playground-cockpit-left-rail-restore')).toBeVisible();
  await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
});
