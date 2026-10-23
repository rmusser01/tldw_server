import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import type { useArtifactsStore } from '../../../packages/ui/src/store/artifacts';

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
    test(`calm header ${viewport.name} ${theme}`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await prepareOfflinePreview(page);
      await page.addInitScript(
        ({ theme, desktop }) => {
          localStorage.setItem('theme', theme);
          localStorage.setItem('playgroundChatContextRailVisible', String(desktop));
          localStorage.setItem('playgroundChatRuntimeRailVisible', String(desktop));
          localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
        },
        { theme, desktop: viewport.name === 'desktop' }
      );
      await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
      await page.setViewportSize(viewport);
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      const input = page.getByTestId('chat-input');
      await expect(input).toBeVisible({ timeout: 60_000 });
      const exitFocus = page.getByRole('button', { name: 'Exit focus', exact: true });
      if (await exitFocus.isVisible()) await exitFocus.click();
      const header = page.getByTestId('chat-conversation-header');
      const view = page.getByRole('group', { name: 'Conversation view', exact: true });
      await expect(header).toBeVisible();
      await expect(header.getByRole('heading')).toBeVisible();
      await expect(
        view.getByRole('button', { name: 'Enter focus chat', exact: true })
      ).toBeVisible();
      await expect(page.getByTestId('chat-header-theme-toggle')).toHaveCount(0);
      await expect(header.getByTestId('chat-header-persistence-badge')).toHaveAttribute(
        'data-persistence-kind',
        'local'
      );
      await expect(
        header.getByRole('button', { name: 'Search commands - Open command palette', exact: true })
      ).toBeVisible();
      await expect(
        header.getByRole('button', { name: 'New saved chat', exact: true })
      ).toBeVisible();
      await input.fill('Keep this draft while I use the header.');
      await expect(page.getByTestId('model-selector')).toBeVisible();
      await page.evaluate(async () => {
        await document.fonts.ready;
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
      });
      expect((await header.boundingBox())!.height).toBeLessThanOrEqual(80);
      expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(
        false
      );
      const axe = await new AxeBuilder({ page })
        .include('[data-testid="chat-conversation-header"]')
        .include('[role="group"][aria-label="Conversation view"]')
        .analyze();
      expect(axe.violations).toEqual([]);
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath(`${viewport.name}-${theme}-header.png`),
      });
      const more = header.getByRole('button', { name: 'More actions', exact: true });
      await more.click();
      const menu = page.getByRole('menu');
      await expect(menu).toBeVisible();
      await expect(
        menu.getByRole('menuitem', { name: 'Open settings', exact: true })
      ).toBeVisible();
      await expect(
        menu.getByRole('menuitem', { name: 'Temporary chat (on this device)', exact: true })
      ).toBeVisible();
      await expect(
        menu.getByRole('menuitem', { name: 'Character chat', exact: true })
      ).toBeVisible();
      expect(await menu.evaluate((node) => node.contains(document.activeElement))).toBe(true);
      if (viewport.name === 'mobile')
        await expect(
          menu.getByRole('menuitem', { name: 'Share conversation', exact: true })
        ).toHaveAttribute('aria-disabled', 'true');
      const menuAxe = await new AxeBuilder({ page }).include('#chat-header-actions-menu').analyze();
      expect(menuAxe.violations).toEqual([]);
      if (
        (viewport.name === 'desktop' && theme === 'dark') ||
        (viewport.name === 'mobile' && theme === 'light')
      ) {
        await page.screenshot({
          animations: 'disabled',
          path: testInfo.outputPath(`${viewport.name}-${theme}-menu.png`),
        });
      }
      await page.keyboard.press('Escape');
      await expect(menu).toBeHidden();
      await expect(more).toBeFocused();
      await expect(input).toHaveValue('Keep this draft while I use the header.');
      await more.click();
      await page.getByRole('menuitem', { name: 'Browse pages', exact: true }).click();
      const launcher = page.getByRole('dialog').last();
      await expect(launcher).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(launcher).toBeHidden();
      await expect(more).toBeFocused();
      await more.click();
      await page.getByRole('menuitem', { name: 'Show keyboard shortcuts', exact: true }).click();
      const shortcuts = page.getByRole('dialog').last();
      await expect(shortcuts).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(shortcuts).toBeHidden();
      await expect(more).toBeFocused();
      await more.click();
      await page
        .getByRole('menuitem', {
          name: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
          exact: true,
        })
        .click();
      await expect(page.locator('html')).toHaveClass(theme === 'dark' ? /light/ : /dark/);
      // The offline history scope can reject the palette on dev; verify the header request here.
      await page.evaluate(() => {
        window.addEventListener('tldw:open-command-palette', () => {
          document.documentElement.dataset.headerPaletteRequested = 'true';
        }, { once: true });
      });
      await header
        .getByRole('button', { name: 'Search commands - Open command palette', exact: true })
        .click();
      await expect(page.locator('html')).toHaveAttribute('data-header-palette-requested', 'true');
      await page.keyboard.press('Escape');
      await expect(input).toHaveValue('Keep this draft while I use the header.');
      await view.getByRole('button', { name: 'Show keyboard shortcuts', exact: true }).click();
      const viewHelp = page.getByRole('dialog').last();
      await expect(viewHelp).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(viewHelp).toBeHidden();
      await expect(page.getByTestId('playground-shortcuts-help-trigger')).toBeFocused();
      await view.getByRole('button', { name: 'Enter focus chat', exact: true }).click();
      await expect(header).toBeHidden();
      await expect(input).toHaveValue('Keep this draft while I use the header.');
      await page.getByRole('button', { name: 'Exit focus', exact: true }).click();
      await expect(header).toBeVisible();
      await more.click();
      await page.getByRole('menuitem', { name: 'Temporary chat (on this device)', exact: true }).click();
      await expect(header.getByTestId('chat-header-persistence-badge')).toHaveAttribute(
        'data-persistence-kind',
        'temporary'
      );
      if (viewport.name === 'desktop')
        await expect(header.getByTestId('chat-header-share-button')).toBeDisabled();
      else {
        await more.click();
        await expect(
          page.getByRole('menuitem', { name: 'Share conversation', exact: true })
        ).toHaveAttribute('aria-disabled', 'true');
        await page.keyboard.press('Escape');
      }
      expect(errors).toEqual([]);
    });
  }
}

test('announces artifact counters in the view control', async ({ page }) => {
  await prepareOfflinePreview(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 60_000 });
  await page.evaluate(() => {
    const artifact = {
      id: 'header-review-artifact',
      title: 'Synthetic analysis',
      kind: 'code' as const,
      content: 'example',
      language: 'text',
    };
    (
      window as unknown as { __tldw_useArtifactsStore: typeof useArtifactsStore }
    ).__tldw_useArtifactsStore.setState({
      active: artifact,
      isOpen: false,
      isPinned: true,
      unreadCount: 2,
      history: [artifact, { ...artifact, id: 'two' }, { ...artifact, id: 'three' }],
    });
  });
  await expect(page.getByTestId('playground-artifacts-trigger')).toHaveAccessibleName(
    /New 2.*Pinned 1.*3 total/
  );
});
