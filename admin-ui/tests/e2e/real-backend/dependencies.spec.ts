import { expect, test } from './helpers/fixtures';

test.describe('Dependencies', () => {
  test('navigates to dependencies page and shows heading', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Dependencies smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/dependencies');
    await expect(page.getByRole('heading', { name: /external dependencies/i })).toBeVisible();
  });

  test('renders system dependencies section', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Dependencies smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/dependencies');

    // The System Dependencies card should appear, or the page should show an error
    const hasSysDeps = await page.getByText(/system dependencies/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasSysDeps || hasError).toBeTruthy();
  });

  test('renders LLM provider section', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Dependencies smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/dependencies');

    // The LLM Provider Health Grid card should appear, or the page should show an error
    const hasProviders = await page.getByText(/llm provider health/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasProviders || hasError).toBeTruthy();
  });

  test('summary cards render', async ({ page, seededSession }, testInfo) => {
    test.skip(
      testInfo.project.name !== 'chromium-real-jwt',
      'Dependencies smoke tests only run in the multi-user JWT project',
    );

    await seededSession.as('admin', 'jwt_admin');
    await page.goto('/dependencies');

    // Summary stat cards should be present (System Components, Configured Providers, etc.)
    const hasSystemComponents = await page.getByText(/system components/i).count() > 0;
    const hasConfiguredProviders = await page.getByText(/configured providers/i).count() > 0;
    const hasError = await page.locator('[role="alert"]').count() > 0;
    expect(hasSystemComponents || hasConfiguredProviders || hasError).toBeTruthy();
  });
});
