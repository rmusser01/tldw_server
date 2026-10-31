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

  test('shows registrations or actionable operational status', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Webhook smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/webhooks');

    await expect(
      page
        .locator('table')
        .or(page.getByText(/no webhooks configured/i))
        .or(page.getByLabel('Webhook operational status'))
        .first(),
    ).toBeVisible();
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
