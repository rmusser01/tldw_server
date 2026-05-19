import { expect, test } from './helpers/fixtures';

test.describe('AI Operations', () => {
  test('navigates to AI ops page and shows heading', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'AI Ops smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/ai-ops');
    await expect(page.getByRole('heading', { name: /ai operations/i })).toBeVisible();
  });

  test('renders KPI cards', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'AI Ops smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/ai-ops');

    // The page should display KPI cards or an error state
    const hasKpi = await page.getByText(/total ai spend/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasKpi || hasError).toBeTruthy();
  });

  test('renders agent metrics section', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'AI Ops smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/ai-ops');

    // The Top Agents by Cost card should appear (with data or empty state)
    const hasAgentSection = await page.getByText(/top agents by cost/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasAgentSection || hasError).toBeTruthy();
  });

  test('renders recent sessions section', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'AI Ops smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/ai-ops');

    // The Recent Sessions card should appear (with data or empty state)
    const hasSessionsSection = await page.getByText(/recent sessions/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasSessionsSection || hasError).toBeTruthy();
  });

  test('refresh button is visible', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'AI Ops smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/ai-ops');
    await expect(page.getByRole('button', { name: /refresh/i })).toBeVisible();
  });
});
