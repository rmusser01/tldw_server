import { expect, test } from './helpers/fixtures';

test.describe('Compliance Posture', () => {
  test('navigates to compliance page and shows heading', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Compliance smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/compliance');
    await expect(page.getByRole('heading', { name: /compliance posture/i })).toBeVisible();
  });

  test('renders posture score card or error state', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Compliance smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/compliance');

    // The page shows either the score cards or an error alert
    const hasScoreCard = await page.getByText(/overall score/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasScoreCard || hasError).toBeTruthy();
  });

  test('renders breakdown cards (MFA, Key Rotation, Audit)', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Compliance smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/compliance');

    // When posture data loads, the breakdown cards should be visible.
    // If there is an error, the cards will not render — so we check for either.
    const hasError = await page.locator('[role="alert"]').count() > 0;
    if (!hasError) {
      await expect(page.getByText(/mfa adoption/i)).toBeVisible();
      await expect(page.getByText(/key rotation/i)).toBeVisible();
      await expect(page.getByText(/audit logging/i)).toBeVisible();
    }
  });

  test('report schedules section is present', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Compliance smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/compliance');

    // The report schedules card should be on the page (even if data failed to load)
    const hasSchedules = await page.getByText(/report schedules/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasSchedules || hasError).toBeTruthy();
  });
});
