import { expect, test } from './helpers/fixtures';

test.describe('Webhook Management', () => {
  test('navigates to webhooks page', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Webhook smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/webhooks');
    await expect(page.locator('h1, [class*="CardTitle"]').filter({ hasText: /webhooks/i }).first()).toBeVisible();
  });

  test('shows empty state or webhook list', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Webhook smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/webhooks');

    // Either the empty state message or a table with webhooks should be visible
    const hasTable = await page.locator('table').count() > 0;
    const hasEmpty = await page.getByText(/no webhooks configured/i).count() > 0;
    expect(hasTable || hasEmpty).toBeTruthy();
  });

  test('add webhook button is visible', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Webhook smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/webhooks');
    await expect(page.getByRole('button', { name: /add webhook/i })).toBeVisible();
  });
});
