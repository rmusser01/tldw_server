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
    test(`calm composer ${viewport.name} ${theme}`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await prepareOfflinePreview(page);
      await page.addInitScript((value) => localStorage.setItem('theme', value), theme);
      await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
      await page.setViewportSize(viewport);
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      const input = page.getByTestId('chat-input');
      await expect(input).toBeVisible({ timeout: 30_000 });
      const switchTheme = page
        .getByRole('button', { name: `Switch to ${theme} theme`, exact: true })
        .first();
      if (await switchTheme.isVisible()) await switchTheme.click();
      await expect(page.locator('html')).toHaveClass(theme === 'dark' ? /dark/ : /light/);
      const tools = page.getByTestId('composer-options-toggle');
      await expect(tools).toHaveAccessibleName('Tools');
      await expect(tools).toHaveAttribute('aria-expanded', 'false');
      for (const control of [
        page.getByTestId('model-selector'),
        page.getByTestId('attachment-button'),
        page.getByRole('button', { name: 'Queue request', exact: true }),
      ]) {
        await expect(control).toBeVisible();
        const box = await control.boundingBox();
        expect(box!.width).toBeGreaterThan(24);
        expect(box!.x).toBeGreaterThanOrEqual(0);
        expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width);
        expect(box!.y + box!.height).toBeLessThanOrEqual(viewport.height);
      }
      const hints = page.getByText('/ commands · @ mentions', { exact: true });
      const saved = page.getByTestId('composer-persistence');
      await expect(hints).toBeVisible();
      await expect(saved).toBeVisible();
      const model = page.getByTestId('model-selector');
      const delivery = page.getByRole('button', { name: 'Queue request', exact: true });
      const rows = viewport.name === 'mobile'
        ? [[tools, hints], [saved, model, delivery]]
        : [[tools, hints, saved, model, delivery]];
      for (const row of rows) {
        const boxes = await Promise.all(row.map(async (control) =>
          (await control.boundingBox())!
        ));
        expect(Math.max(...boxes.map((box) => box.y))).toBeLessThan(
          Math.min(...boxes.map((box) => box.y + box.height))
        );
      }
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath(`${viewport.name}-${theme}-closed.png`),
      });
      await input.fill('Keep this draft while I choose a tool.');
      await tools.click();
      const panel = page.getByRole('region', { name: 'Composer tools' });
      await expect(panel).toBeVisible();
      await expect(
        page.getByRole('button', { name: 'Search & Context', exact: true })
      ).toBeVisible();
      await expect(
        page.getByRole('button', { name: 'Advanced controls', exact: true })
      ).toHaveCount(0);
      await expect(input).toHaveValue('Keep this draft while I choose a tool.');
      const relatedRows = [
        [model, page.getByTestId('dictation-button')],
        [page.getByTestId('chat-prompt-select'), panel.getByRole('button', { name: 'System prompts', exact: true })],
        [panel.getByRole('button', { name: 'Chat Settings', exact: true }), panel.getByRole('button', { name: 'Generation style', exact: true })],
        [panel.getByTestId('composer-formatting-guide-toggle'), panel.getByTestId('openui-request-mode-toggle')],
        [panel.getByTestId('knowledge-search-toggle'), panel.getByTestId('web-search-toggle')],
      ];
      for (const row of relatedRows) {
        const boxes = await Promise.all(row.map(async (control) =>
          (await control.boundingBox())!
        ));
        expect(Math.max(...boxes.map((box) => box.y))).toBeLessThan(
          Math.min(...boxes.map((box) => box.y + box.height))
        );
      }
      const accessibility = await new AxeBuilder({ page })
        .include('[data-testid="composer-toolbar"]')
        .analyze();
      expect(accessibility.violations).toEqual([]);
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath(`${viewport.name}-${theme}-tools.png`),
      });
      await page.keyboard.press('Escape');
      await expect(panel).toBeHidden();
      await expect(tools).toBeFocused();
      expect(errors).toEqual([]);
    });
  }
}

test('remembers whether Tools was expanded across reload', async ({ page }) => {
  await prepareOfflinePreview(page);
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  const tools = page.getByTestId('composer-options-toggle');
  await expect(tools).toHaveAttribute('aria-expanded', 'false');
  await tools.click();
  await expect(tools).toHaveAttribute('aria-expanded', 'true');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(tools).toHaveAttribute('aria-expanded', 'true');
});

for (const variant of ['v1', 'v3', 'v5']) {
  test(`keeps one delivery control in the optional ${variant} shell`, async ({ page }) => {
    await prepareOfflinePreview(page);
    await page.addInitScript(
      (value) => localStorage.setItem('tldw:composerVariant', value),
      variant
    );
    await page.goto('/chat?nextgenComposer=1', { waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('nextgen-composer-wrapper')).toBeVisible();
    await expect(page.getByTestId('chat-input')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Queue request', exact: true })).toHaveCount(1);
    await expect(page.getByTestId('model-selector')).toBeVisible();
  });
}
