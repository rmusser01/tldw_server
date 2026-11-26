import { expect, test, type Page } from '@playwright/test';
import path from 'node:path';
import { installBrowserFixture } from './utils/media-ux-fixture';

const destination = (page: Page) =>
  page.getByRole('combobox', { name: 'Destination', exact: true });
const preset = (page: Page, name: string) => page.getByRole('radio', { name, exact: true });

async function captureSettings(page: Page, filename: string) {
  const dir = process.env.MEDIA_UX_EVIDENCE_DIR;
  if (dir) {
    await page.evaluate(() => document.fonts.ready.then(() => undefined));
    await page.screenshot({ path: path.join(dir, filename), animations: 'disabled' });
  }
}

test.describe('Quick Ingest preset settings', () => {
  test.setTimeout(120_000);

  test('a fresh import sends the destination and analysis provider saved in Standard', async ({
    page,
  }) => {
    const fixture = await installBrowserFixture(page, { empty: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto('/settings/quick-ingest', { waitUntil: 'domcontentloaded' });
    const editor = page.getByRole('region', { name: /Standard preset/i });
    await expect(preset(page, 'Standard')).toBeChecked({ timeout: 60_000 });
    const provider = editor.getByRole('combobox', { name: 'Analysis provider', exact: true });
    await expect(provider).toHaveValue('');
    await expect(provider).toHaveAccessibleDescription(/turn off Analysis before/i);
    await expect(editor.getByRole('switch', { name: 'Analysis', exact: true })).toBeChecked();
    const prerequisite = editor.getByText(/turn off Analysis before/i);
    await expect(prerequisite).toBeVisible();
    await captureSettings(page, 'desktop.png');

    await destination(page).selectOption({ label: 'Extract without saving' });
    await provider.fill('openai');
    await provider.press('Tab');
    await expect(prerequisite).toBeHidden();
    await page.reload();
    await expect(provider).toHaveValue('openai');
    await expect(destination(page).locator('option:checked')).toHaveText('Extract without saving');

    await page.goto('/media', { waitUntil: 'domcontentloaded' });
    const source = 'https://example.com/preset-settings-test';
    await page.getByPlaceholder('Paste a YouTube URL...').fill(source);
    await page.getByRole('button', { name: 'Ingest', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: /Quick Ingest/i });
    await dialog.getByRole('button', { name: /Use defaults.*process|^Extract 1 source$/i }).click();
    await expect
      .poll(
        () =>
          fixture.requests.find(
            (request) =>
              request.method === 'POST' && request.path === '/api/v1/media/process-web-scraping'
          )?.body
      )
      .toMatchObject({
        url_input: source,
        mode: 'ephemeral',
        summarize_checkbox: true,
        api_name: 'openai',
      });
  });

  test('phone users can reach the destination, change presets by keyboard and undo resets', async ({
    page,
  }) => {
    await installBrowserFixture(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/settings/quick-ingest', { waitUntil: 'domcontentloaded' });
    const presets = page.getByRole('radiogroup', { name: 'Preset to edit', exact: true });
    const standard = presets.getByRole('radio', { name: 'Standard', exact: true });
    await expect(standard).toBeChecked({ timeout: 60_000 });
    await expect(destination(page)).toBeInViewport();
    await expect(destination(page).locator('option')).toHaveText([
      'Save to Media',
      'Create review drafts',
      'Extract without saving',
    ]);
    const fits = await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth
    );
    expect(fits, 'the 390px settings page fits horizontally').toBe(true);
    await captureSettings(page, 'phone.png');

    await standard.focus();
    await standard.press('ArrowLeft');
    await expect(preset(page, 'Quick')).toBeChecked();
    await destination(page).selectOption({ label: 'Extract without saving' });
    await preset(page, 'Quick').press('ArrowRight');
    await expect(standard).toBeChecked();
    await destination(page).selectOption({ label: 'Create review drafts' });
    await page.getByRole('button', { name: 'Reset Standard preset', exact: true }).click();
    await expect(destination(page).locator('option:checked')).toHaveText('Save to Media');
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    await expect(destination(page).locator('option:checked')).toHaveText('Create review drafts');

    await page.getByRole('button', { name: 'Reset all presets', exact: true }).click();
    await expect(destination(page).locator('option:checked')).toHaveText('Save to Media');
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    await expect(destination(page).locator('option:checked')).toHaveText('Create review drafts');
    await standard.press('ArrowLeft');
    await expect(preset(page, 'Quick')).toBeChecked();
    await expect(destination(page).locator('option:checked')).toHaveText('Extract without saving');
  });
});
