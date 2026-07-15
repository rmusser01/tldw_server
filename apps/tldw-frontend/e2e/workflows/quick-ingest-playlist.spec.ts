import AxeBuilder from '@axe-core/playwright';

import { test, expect, assertNoCriticalErrors } from '../utils/fixtures';
import { waitForConnection } from '../utils/helpers';
import { openQuickIngestDialog, reopenQuickIngest } from '../utils/journey-helpers';
import {
  CONFERENCE_PLAYLIST_URL,
  installQuickIngestPlaylistV2Routes,
} from '../fixtures/quick-ingest-playlist-v2';

test.describe('Quick Ingest playlist v2', () => {
  test('shows every playlist occurrence and reattaches one bounded run after reload', async ({
    authedPage,
    diagnostics,
  }) => {
    test.setTimeout(180_000);
    await authedPage.addInitScript(() => {
      localStorage.removeItem('tldw-quick-ingest-session');
      localStorage.removeItem('__tldwServerCapabilitiesCacheV3');
      sessionStorage.removeItem('tldw-quick-ingest-session');
      sessionStorage.removeItem('__tldwServerCapabilitiesCacheV3');
    });
    const fixture = await installQuickIngestPlaylistV2Routes(authedPage);
    const docsInfoReady = authedPage.waitForResponse(
      (response) => response.url().includes('/api/v1/config/docs-info') && response.ok()
    );
    const openApiReady = authedPage.waitForResponse(
      (response) => response.url().endsWith('/openapi.json') && response.ok()
    );

    await authedPage.goto('/media', { waitUntil: 'domcontentloaded' });
    await Promise.all([waitForConnection(authedPage), docsInfoReady, openApiReady]);
    let dialog = await openQuickIngestDialog(authedPage);
    const urlInput = dialog.getByLabel('Paste URLs input');

    await urlInput.fill(CONFERENCE_PLAYLIST_URL);
    await dialog.getByRole('button', { name: 'Add URLs' }).click();

    await expect(dialog).toContainText('Conference 2010', { timeout: 20_000 });
    await expect(dialog).toContainText('34 items');
    await expect(dialog).toContainText('32 selected');
    await expect(dialog).toContainText('2 duplicates');
    const playlist = dialog.getByRole('list', { name: 'Playlist videos' });
    await expect(playlist.getByRole('listitem').first()).toHaveAttribute('aria-setsize', '34');
    await dialog.getByRole('checkbox', { name: 'Select playlist item 3: Talk 3' }).uncheck();
    await expect(dialog).toContainText('31 selected');

    const accessibility = await new AxeBuilder({ page: authedPage })
      .include('[role="dialog"]')
      .analyze();
    expect(
      accessibility.violations.filter(
        (violation) => violation.impact === 'critical' || violation.impact === 'serious'
      )
    ).toEqual([]);
    expect(fixture.getRemoteThumbnailRequestCount()).toBe(0);

    await expect
      .poll(
        async () => {
          await playlist.evaluate((element) => {
            element.scrollTop = element.scrollHeight;
            element.dispatchEvent(new Event('scroll', { bubbles: true }));
          });
          return playlist.getByText('34. Talk 34').isVisible();
        },
        { timeout: 15_000 }
      )
      .toBe(true);

    await dialog.getByRole('button', { name: 'Add 31 videos' }).click();
    await expect(dialog.getByRole('button', { name: /configure 31 items/i })).toBeVisible();
    expect(fixture.getMaterializedOccurrenceIds()).toHaveLength(31);

    await dialog.getByRole('button', { name: /configure 31 items/i }).click();
    await dialog.getByRole('button', { name: 'Next' }).click();
    await expect(dialog).toContainText('Ready to Process');
    await dialog.getByRole('button', { name: /start processing/i }).click();

    await expect(dialog.getByRole('heading', { name: 'Processing' })).toBeVisible({
      timeout: 30_000,
    });
    const processingItems = dialog.getByRole('list', {
      name: 'Processing items',
    });
    await expect(processingItems.getByRole('listitem').first()).toHaveAttribute(
      'aria-setsize',
      '31'
    );
    await expect(dialog).toContainText(/processing|running/i);
    await expect.poll(() => fixture.getRunInputs().length, { timeout: 30_000 }).toBe(31);
    await expect.poll(() => fixture.getSubmissionPostCount(), { timeout: 30_000 }).toBe(1);
    await expect(processingItems.getByRole('listitem').first()).toContainText('Running', {
      timeout: 30_000,
    });
    await expect(processingItems.getByRole('listitem').first()).toContainText('45%');
    await expect
      .poll(async () => (await fixture.getDurableTracking())?.runId, {
        timeout: 30_000,
      })
      .toBe('run-conference-v2');
    expect(fixture.getRunInputs()).toHaveLength(31);
    expect(
      fixture
        .getRunInputs()
        .every(
          (input) =>
            input.input_kind === 'materialized_playlist_item' &&
            input.materialization_id === 'materialization-conference-v2'
        )
    ).toBe(true);
    expect(fixture.getSubmissionPostCount()).toBe(1);

    const runItemsRequestsBeforeReload = fixture.getRunItemsRequestCount();
    const runSummaryRequestsBeforeReload = fixture.getRunSummaryRequestCount();
    await authedPage.reload({ waitUntil: 'domcontentloaded' });
    await waitForConnection(authedPage);
    dialog = await reopenQuickIngest(authedPage);
    await expect(dialog.getByRole('heading', { name: 'Processing' })).toBeVisible({
      timeout: 30_000,
    });
    const reattachedProcessingItem = dialog
      .getByRole('list', { name: 'Processing items' })
      .getByRole('listitem')
      .first();
    await expect(reattachedProcessingItem).toHaveAttribute('aria-setsize', '31');
    await expect
      .poll(() => fixture.getRunSummaryRequestCount(), { timeout: 30_000 })
      .toBeGreaterThan(runSummaryRequestsBeforeReload);
    await expect
      .poll(() => fixture.getRunItemsRequestCount(), { timeout: 30_000 })
      .toBeGreaterThan(runItemsRequestsBeforeReload);
    await expect
      .poll(async () => (await fixture.getDurableFirstProgress())?.lifecycleState, {
        timeout: 30_000,
      })
      .toBe('running');
    await expect(reattachedProcessingItem).toContainText('Running', {
      timeout: 30_000,
    });
    await expect(reattachedProcessingItem).toContainText('45%');
    await expect(dialog).toContainText(/processing|running/i);

    fixture.completeRun();
    await expect(dialog.getByTestId('wizard-results-step')).toBeVisible({
      timeout: 30_000,
    });
    await expect(dialog).toContainText('Completed (30)', { timeout: 30_000 });
    await expect(dialog).toContainText('Failed during processing (1)');
    await expect(dialog).toContainText('Total: 30 succeeded, 1 failed');

    expect(fixture.getPreflightItemsRequestCount()).toBe(2);
    expect(fixture.getSubmissionPostCount()).toBe(1);
    expect(fixture.getPerItemPollCount()).toBe(0);
    expect(fixture.getRemoteThumbnailRequestCount()).toBe(0);
    await assertNoCriticalErrors(diagnostics);
  });
});
