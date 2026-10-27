import { test, expect, type Page } from '@playwright/test';
import path from 'node:path';
import { installBrowserFixture, FRONTEND_URL } from '../utils/media-ux-fixture';

const evidence = process.env.MEDIA_UX_EVIDENCE_DIR;
async function screenshot(page: Page, name: string) {
  if (evidence) await page.screenshot({ path: path.join(evidence, `${name}.png`) });
}
async function media(page: Page) {
  await page.goto(`${FRONTEND_URL}/media`);
  await expect(page.getByRole('button', { name: /Select media:/ }).first()).toBeVisible();
}

test.describe('Complete Media UX against isolated simulated outcomes', () => {
  test.setTimeout(120_000);
  test.use({ actionTimeout: 15_000 });

  test('empty library hands off its first URL and resumes a minimized active import', async ({
    page,
  }) => {
    const state = await installBrowserFixture(page, { empty: true, processingPolls: 50 });
    await page.goto(`${FRONTEND_URL}/media`);
    const source = 'https://example.com/first-native-source';
    await page.getByPlaceholder('Paste a YouTube URL...').fill(source);
    await page.getByRole('button', { name: 'Ingest', exact: true }).click();
    const wizard = page.getByRole('dialog', { name: 'Quick Ingest' });
    await expect(wizard.getByText(source, { exact: true }).first()).toBeVisible();
    await wizard.locator('input[type=file]').setInputFiles({
      name: 'resume.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('safe simulated resume source'),
    });
    await wizard.getByRole('button', { name: /Configure.*2/ }).click();
    await wizard.getByRole('button', { name: /^Quick/ }).click();
    await wizard.getByRole('button', { name: 'Next', exact: true }).click();
    await wizard.getByRole('button', { name: 'Start processing', exact: true }).click();
    await wizard.getByRole('button', { name: 'Minimize to Background', exact: true }).click();
    await expect(wizard).not.toBeVisible();
    await page.getByTestId('media-ingest-jobs-toggle').click();
    await page.getByRole('button', { name: 'Resume import', exact: true }).click();
    await expect(wizard).toBeVisible();
    await expect(
      wizard.getByRole('button', { name: 'Minimize to Background', exact: true })
    ).toBeVisible();
    state.processingPolls = 0;
    for (const job of state.jobs.values()) job.polls = 0;
    await expect(
      wizard.getByRole('status').filter({ hasText: /added.*excluded\/skipped/ })
    ).toContainText('2 succeeded (2 saved)', { timeout: 45000 });
    await screenshot(page, 'webui-empty-first-source-resumed');
    await wizard.getByRole('button', { name: 'Review these 2 saved items', exact: true }).click();
    await expect(wizard).not.toBeVisible();
    await expect(page.getByTestId('media-review-reading-window')).toContainText('of 2 selected');
    await expect(
      page.getByRole('button', { name: 'Chat about selection (2)', exact: true })
    ).toBeEnabled();
  });

  test('mixed import reconciles exclusions, retries only the failure and restores recognizable history', async ({
    page,
  }) => {
    const state = await installBrowserFixture(page);
    await media(page);
    await page.getByRole('button', { name: 'Add media', exact: true }).click();
    const wizard = page.getByRole('dialog', { name: 'Quick Ingest' });
    await wizard
      .getByRole('textbox', { name: 'Paste URLs input' })
      .fill(
        'https://example.com/saved-one, https://example.com/fail-once\nhttps://example.com/saved-one\ninvalid\nhttps://example.com/a,b'
      );
    await wizard.getByRole('button', { name: 'Add URLs to queue', exact: true }).click();
    await wizard.locator('input[type=file]').setInputFiles({
      name: 'evidence.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('isolated simulated evidence'),
    });
    await wizard.getByRole('button', { name: /Configure.*4/ }).click();
    await wizard.getByRole('button', { name: /^Quick/ }).click();
    await wizard.getByRole('button', { name: 'Next', exact: true }).click();
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(wizard.getByText(/Replacement disabled:/)).toBeVisible();
    await screenshot(page, 'webui-mobile-review');
    await wizard.getByRole('button', { name: 'Start processing', exact: true }).click();
    await expect(
      wizard.getByRole('status').filter({ hasText: /added.*excluded\/skipped/ })
    ).toContainText('1 failed', { timeout: 45_000 });
    await expect(
      wizard.getByRole('status').filter({ hasText: /added.*excluded\/skipped/ })
    ).toContainText('3 succeeded (3 saved)');
    await expect(
      wizard.getByRole('status').filter({ hasText: /added.*excluded\/skipped/ })
    ).toContainText('2 excluded/skipped');
    await wizard
      .getByRole('button', { name: 'Retry https://example.com/fail-once', exact: true })
      .click();
    await expect(
      wizard.getByRole('status').filter({ hasText: /added.*excluded\/skipped/ })
    ).toContainText('4 succeeded (4 saved)', {
      timeout: 45_000,
    });
    expect(state.attempts.get('https://example.com/saved-one')).toBe(1);
    expect(state.attempts.get('https://example.com/a,b')).toBe(1);
    expect(state.attempts.get('https://example.com/fail-once')).toBe(2);
    expect(
      state.requests.filter((r) => r.method === 'POST' && r.path === '/api/v1/media/ingest/jobs')
    ).toHaveLength(1);
    await expect(wizard.getByText('Knowledge readiness unconfirmed')).toHaveCount(4);
    await screenshot(page, 'webui-mobile-results');
    await wizard.getByRole('button', { name: 'Close the ingest wizard', exact: true }).click();
    await page.reload();
    await page.getByTestId('media-ingest-jobs-toggle').click();
    const history = page.getByTestId('media-ingest-jobs-panel');
    await expect(history.getByText('example.com', { exact: true })).toBeVisible();
    await expect(page.getByTestId('media-library-tools-toggle')).toHaveAttribute(
      'aria-expanded',
      'false'
    );
    await history.getByRole('button', { name: 'Refresh import', exact: true }).click();
    await expect
      .poll(
        () =>
          state.requests.filter((r) => r.method === 'GET' && r.path === '/api/v1/media/ingest/jobs')
            .length
      )
      .toBeGreaterThan(0);
    await history.getByRole('button', { name: 'Review 4 saved items', exact: true }).click();
    await expect(page.getByTestId('media-review-reading-window')).toContainText('of 4 selected');
    await screenshot(page, 'webui-mobile-saved-set');
  });

  test('Inspector preserves cross-page selection with mobile reading, history and reversible trash', async ({
    page,
  }) => {
    const state = await installBrowserFixture(page);
    await media(page);
    // Use a real verified draft to seed ten owner-scoped historical metadata rows.
    await page.getByRole('button', { name: 'Add media', exact: true }).click();
    await page
      .getByRole('dialog', { name: 'Quick Ingest' })
      .getByRole('button', { name: 'Close', exact: true })
      .click();
    await page.evaluate(() => {
      const key = 'tldw-quick-ingest-session';
      const value = JSON.parse(sessionStorage.getItem(key)!);
      const authorityKey = value.state.session.authorityKey;
      value.state.recentImports = Array.from({ length: 10 }, (_, i) => ({
        id: `geometry-history-${i}`,
        authorityKey,
        sourceLabel: `Research import ${i + 1}`,
        sourceCount: 4,
        lifecycle: 'completed',
        createdAt: Date.now() - i * 60000,
        updatedAt: Date.now() - i * 60000,
        completedAt: Date.now() - i * 60000,
        batchIds: [],
        jobIds: [],
        savedMediaIds: [1, 2, 3, 4],
      }));
      sessionStorage.setItem(key, JSON.stringify(value));
    });
    await page.reload();
    await page.getByTestId('media-ingest-jobs-toggle').click();
    await expect(page.getByTestId('media-ingest-jobs-panel').getByRole('listitem')).toHaveCount(10);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: /Select media: Understanding retrieval/ }).click();
    await expect(page.getByRole('button', { name: 'Back to results', exact: true })).toBeVisible();
    await screenshot(page, 'webui-mobile-content');
    await page.getByRole('button', { name: 'Back to results', exact: true }).click();
    await page.getByTestId('media-bulk-mode-toggle').click();
    await page.getByTestId('results-select-1').check();
    await page.getByTestId('results-select-2').check();
    await page.getByRole('button', { name: 'Next page', exact: true }).click();
    await page.getByTestId('results-select-21').check();
    await page.getByTestId('results-select-22').check();
    await expect(page.getByTestId('media-bulk-open-multi')).toBeVisible();
    await expect(page.getByTestId('media-bulk-delete')).toHaveText(/Move 4 items to trash/);
    await expect(page.getByTestId('media-ingest-jobs-toggle')).toHaveAttribute(
      'aria-expanded',
      'true'
    );
    await expect(page.getByTestId('media-bulk-open-multi')).toBeInViewport();
    await screenshot(page, 'webui-mobile-selection-history');
    await page.setViewportSize({ width: 844, height: 390 });
    await expect(page.getByTestId('media-bulk-open-multi')).toBeInViewport();
    await page.getByTestId('media-bulk-open-multi').click({ trial: true });
    await page.getByRole('button', { name: 'Previous page', exact: true }).click();
    await expect(page.getByTestId('results-select-1')).toBeChecked();
    await page.getByRole('button', { name: 'Next page', exact: true }).click();
    await expect(page.getByTestId('results-select-21')).toBeChecked();
    await expect(page.getByTestId('media-bulk-open-multi')).toBeInViewport();
    await screenshot(page, 'webui-landscape-selection');
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.evaluate(() => {
      document.documentElement.style.fontSize = '160%';
    });
    await expect(page.getByTestId('media-bulk-open-multi')).toBeInViewport();
    await page.getByTestId('media-bulk-open-multi').click({ trial: true });
    await screenshot(page, 'webui-large-text-selection');
    await page.evaluate(() => {
      document.documentElement.style.fontSize = '';
    });
    await page.getByTestId('media-bulk-delete').click();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    expect(state.items.filter((item) => item.is_deleted)).toHaveLength(0);
    await page.getByTestId('media-bulk-delete').click();
    await page.getByRole('button', { name: 'Move to trash', exact: true }).click();
    await expect.poll(() => state.items.filter((item) => item.is_deleted).length).toBe(4);
    await page.getByRole('button', { name: 'Open Trash', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Restore selected', exact: true })).toBeVisible();
    await page.getByRole('checkbox', { name: 'Select all visible', exact: true }).check();
    await page.getByRole('button', { name: 'Restore selected', exact: true }).click();
    await expect.poll(() => state.items.filter((item) => item.is_deleted).length).toBe(0);
    await screenshot(page, 'webui-trash-recovery');
  });

  test('unfiltered Multi-review aligns row keyboard preview and bounds reading of a 40-item metadata set', async ({
    page,
  }) => {
    const state = await installBrowserFixture(page);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(`${FRONTEND_URL}/media-multi`);
    await expect(page.getByTestId('media-review-result-row').first()).toBeVisible();
    await expect(page.getByRole('combobox', { name: 'Sort', exact: true })).toBeVisible();
    const first = page.getByTestId('media-review-result-row').first();
    await first.click();
    await expect(page.getByTestId('media-review-reading-context')).toContainText(
      'Understanding retrieval'
    );
    await first.focus();
    await first.press('Enter');
    await expect(page.getByTestId('media-review-reading-context')).toContainText(
      'Understanding retrieval'
    );
    await expect(
      page.getByRole('checkbox', {
        name: 'Select Understanding retrieval: a field guide',
        exact: true,
      })
    ).not.toBeChecked();
    await page.getByRole('heading', { name: /^Results/, exact: false }).click();
    await page.keyboard.press('ControlOrMeta+a');
    await expect(
      page.getByRole('button', { name: 'Review selected (20)', exact: true })
    ).toBeVisible();
    await page.getByRole('listitem', { name: 'Next Page', exact: true }).click();
    await expect(
      page.getByRole('checkbox', { name: 'Select Research source 21', exact: true })
    ).toBeVisible();
    await page.getByRole('heading', { name: /^Results/, exact: false }).click();
    await page.keyboard.press('ControlOrMeta+a');
    await expect(
      page.getByRole('button', { name: 'Review selected (40)', exact: true })
    ).toBeVisible();
    await page.getByRole('button', { name: 'Review selected (40)', exact: true }).click();
    await expect(page.getByTestId('media-review-reading-window')).toContainText(
      'Reading 1–30 of 40 selected (30 at a time)'
    );
    const detailRequests = () =>
      state.requests.filter((r) => r.method === 'GET' && /^\/api\/v1\/media\/\d+$/.test(r.path));
    await expect.poll(() => detailRequests().length).toBe(30);
    await expect(page.locator('[data-testid^=media-review-content-body-]').first()).toBeVisible();
    expect(
      await page.locator('[data-testid^=media-review-content-body-]').count()
    ).toBeLessThanOrEqual(30);
    await screenshot(page, 'webui-desktop-reading-40');
    await page.getByRole('button', { name: 'Next reading window', exact: true }).click();
    await expect(page.getByTestId('media-review-reading-window')).toContainText(
      'Reading 31–40 of 40 selected'
    );
    await expect.poll(() => detailRequests().length).toBe(40);
    await expect(page.locator('[data-testid^=media-review-content-body-]').first()).toBeVisible();
    expect(
      await page.locator('[data-testid^=media-review-content-body-]').count()
    ).toBeLessThanOrEqual(10);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Content', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Back to results', exact: true })).toBeVisible();
    await screenshot(page, 'webui-mobile-reading-window');
  });
});
