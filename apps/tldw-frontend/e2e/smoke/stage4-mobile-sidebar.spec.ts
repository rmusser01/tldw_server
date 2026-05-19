import { test, expect, seedAuth } from './smoke.setup';
import { waitForAppShell, waitForConnection } from '../utils/helpers';

const LOAD_TIMEOUT = 30_000;

test.use({
  viewport: { width: 375, height: 812 },
});

test.describe('Stage 4 mobile sidebar behavior', () => {
  test('chat sidebar is hidden by default on mobile and supports accessible open/keyboard close', async ({
    page,
  }) => {
    await seedAuth(page);

    await page.goto('/chat', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT,
    });
    await waitForConnection(page, LOAD_TIMEOUT);
    await waitForAppShell(page, LOAD_TIMEOUT);

    const headerToggle = page.getByTestId('chat-header-sidebar-toggle');
    await expect(headerToggle).toBeVisible({ timeout: LOAD_TIMEOUT });
    await expect(headerToggle).toHaveAccessibleName(/expand sidebar/i);
    await expect(page.getByTestId('chat-sidebar-toggle')).toHaveCount(0);

    const sidebar = page.getByTestId('chat-sidebar');
    await expect(sidebar).toHaveCount(0);

    await headerToggle.click();

    await expect(sidebar).toBeVisible({ timeout: LOAD_TIMEOUT });
    await expect(page.getByTestId('chat-sidebar-search')).toBeVisible({
      timeout: LOAD_TIMEOUT,
    });
    await expect(headerToggle).toHaveAccessibleName(/collapse sidebar/i);

    const sidebarCollapseToggle = page.getByTestId('chat-sidebar-toggle');
    await expect(sidebarCollapseToggle).toBeVisible({ timeout: LOAD_TIMEOUT });
    await sidebarCollapseToggle.press('Enter');

    await expect(page.getByRole('dialog', { name: /chats/i })).toHaveCount(0);
    await expect(page.getByTestId('chat-sidebar-search')).not.toBeVisible();
    await expect(headerToggle).toHaveAccessibleName(/expand sidebar/i);
  });
});
