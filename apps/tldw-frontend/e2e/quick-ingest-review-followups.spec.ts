import { expect, test, type Locator, type Page, type TestInfo } from '@playwright/test';
import { installBrowserFixture } from './utils/media-ux-fixture';

const webUrl = new URL(process.env.TLDW_WEB_URL || 'http://localhost:8080').origin;
const existing = 'https://example.org/archive/estuary-report';
const saved = 'https://example.org/archive/survey-notes';
const rejected = 'https://example.org/archive/field-observations/' +
  'november-estuary-bird-count-from-the-north-bank-and-southern-tidal-flats';
const crossOrigin = { 'access-control-allow-origin': '*' };
test.setTimeout(120_000);

async function openIngest(page: Page) {
  await page.goto(webUrl + '/media', { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Add media', exact: true }).click();
  return page.getByRole('dialog', { name: 'Quick Ingest', exact: true });
}

async function fillLinks(dialog: Locator, links: string[]) {
  await dialog.getByRole('button', { name: 'Links', exact: true }).click();
  await dialog.getByRole('textbox', { name: /URLs input/i }).fill(links.join('\n'));
}

async function quickPreset(dialog: Locator) {
  await dialog.getByRole('button', { name: 'Change settings', exact: true }).click();
  await dialog.getByLabel('Processing preset').selectOption('quick');
}

async function screenshot(page: Page, info: TestInfo, name: string) {
  const path = info.outputPath(name + '.png');
  await page.screenshot({ path, animations: 'disabled' });
  await info.attach(name, { path, contentType: 'image/png' });
}

for (const [device, width, height] of [
  ['desktop', 1280, 800], ['mobile', 390, 844], ['narrow', 320, 720],
] as const) {
  test.describe(device, () => {
    test.use({ viewport: { width, height }, isMobile: width < 600, hasTouch: width < 600 });

    test('mixed receipt puts the repair before exclusions and preserves both duplicate reasons', async ({ page }, info) => {
      const fixture = await installBrowserFixture(page, { webUrl, count: 2 });
      const requests: string[] = [];
      await page.route('**/api/v1/media/process-web-scraping', async route => {
        if (route.request().method() !== 'POST') return route.fallback();
        const source = String(route.request().postDataJSON().url_input);
        requests.push(source);
        if (source === rejected) return route.fulfill({
          status: 422, headers: crossOrigin, json: { detail: 'The archive rejected this link.' },
        });
        if (source === existing) return route.fulfill({
          headers: crossOrigin, json: { status: 'skipped', media_ids: [1], db_message: 'Already exists in Media.' },
        });
        return route.fallback();
      });
      const dialog = await openIngest(page);
      await fillLinks(dialog, [saved, existing, existing, rejected]);
      await dialog.getByRole('button', { name: /add urls/i }).click();
      await quickPreset(dialog);
      await dialog.getByRole('button', { name: 'Save 3 sources', exact: true }).click();

      const repair = dialog.getByRole('button', { name: /Edit link/i });
      await expect(repair).toBeVisible();
      const errors = dialog.getByRole('region', { name: /Error items|Not submitted items/i }).filter({ hasText: rejected });
      const local = dialog.getByRole('region', { name: 'Excluded inputs' });
      const library = dialog.getByRole('region', { name: 'Skipped items' });
      await expect(local).toContainText('Repeated in this import');
      await expect(library).toContainText('Already in Media');
      await expect(local).toContainText(existing);
      await expect(library).toContainText(existing);
      const resultOrder = await dialog.locator('section[aria-label]').evaluateAll(sections =>
        sections.map(section => section.getAttribute('aria-label')));
      const failureIndex = resultOrder.indexOf(await errors.getAttribute('aria-label'));
      expect(failureIndex).toBeGreaterThanOrEqual(0);
      expect(failureIndex).toBeLessThan(resultOrder.indexOf('Excluded inputs'));
      expect(failureIndex).toBeLessThan(resultOrder.indexOf('Skipped items'));
      expect(requests.sort()).toEqual([existing, rejected, saved].sort());
      expect(fixture.items).toHaveLength(3);

      if (width < 600) {
        const sourceLines = await errors.getByText(rejected, { exact: true }).evaluate(element => {
          const range = document.createRange();
          range.selectNodeContents(element);
          return [...range.getClientRects()].map(rect => ({ y: rect.y, left: rect.left, right: rect.right }));
        });
        expect(new Set(sourceLines.map(line => line.y)).size).toBeGreaterThan(1);
        for (const line of sourceLines) {
          expect(line.left).toBeGreaterThanOrEqual(0);
          expect(line.right).toBeLessThanOrEqual(width + 1);
        }
        const actions = [repair, library.getByRole('button', { name: /Open/i })];
        for (const action of actions) {
          const bounds = await action.boundingBox();
          expect(bounds?.height).toBeGreaterThanOrEqual(44);
          expect(bounds?.width).toBeGreaterThanOrEqual(44);
        }
      }
      if (device !== 'narrow') await screenshot(page, info, 'mixed-receipt-' + device);
      await repair.click();
      await expect(dialog.getByRole('textbox', { name: 'Correct link' })).toBeFocused();
    });

    if (device === 'narrow') return;

    test('starts a new attempt for just the server-cancelled source', async ({ page }, info) => {
      const fixture = await installBrowserFixture(page, { webUrl, processingPolls: 100 });
      const stopped = 'https://example.org/estuary/cancelled.pdf';
      const completed = 'https://example.org/estuary/saved.pdf';
      const dialog = await openIngest(page);
      await fillLinks(dialog, [completed, stopped]);
      await dialog.getByRole('button', { name: /add urls/i }).click();
      await quickPreset(dialog);
      await dialog.getByRole('button', { name: 'Save 2 sources', exact: true }).click();
      await expect.poll(() => fixture.jobs.size).toBe(2);
      for (const job of fixture.jobs.values()) {
        if (job.source === stopped) job.status = 'cancelled';
        else job.polls = 0;
      }
      const next = dialog.getByRole('region', { name: 'Cancelled items' })
        .getByRole('button', { name: 'Start a new attempt for ' + stopped, exact: true });
      await expect(next).toBeVisible({ timeout: 30_000 });
      await expect(dialog.getByText('Only this source will run again.', { exact: false })).toBeVisible();
      await next.scrollIntoViewIfNeeded();
      await screenshot(page, info, 'cancelled-source-' + device);
      fixture.processingPolls = 0;
      await next.focus();
      await next.press('Enter');
      await expect(dialog.getByRole('heading', { name: '2 saved to Media' })).toBeVisible({ timeout: 30_000 });
      expect(fixture.attempts.get(completed)).toBe(1);
      expect(fixture.attempts.get(stopped)).toBe(2);
      expect(fixture.jobs.size).toBe(3);
    });

    test('progress leads with the completed count and active file and discloses job tracking', async ({ page }, info) => {
      const fixture = await installBrowserFixture(page, { webUrl, count: 2, processingPolls: 100 });
      const fileName = 'North bank field notes.txt';
      const dialog = await openIngest(page);
      await fillLinks(dialog, [saved]);
      await dialog.getByRole('button', { name: /add urls/i }).click();
      await dialog.getByRole('button', { name: 'Files', exact: true }).click();
      await dialog.getByTestId('qi-file-input').setInputFiles({
        name: fileName, mimeType: 'text/plain', buffer: Buffer.from('North bank water level survey.'),
      });
      await quickPreset(dialog);
      await dialog.getByRole('button', { name: 'Save 2 sources' }).click();
      await expect(dialog.getByRole('heading', { name: '1 of 2 finished' })).toBeVisible();
      await expect(dialog.getByRole('listitem').first()).toContainText(fileName);
      await expect(dialog.getByRole('listitem').first()).toContainText('40%');
      const tracking = dialog.locator('details').filter({ has: page.getByText('Tracking details', { exact: true }) });
      await expect(tracking).toHaveJSProperty('open', false);
      const ids = tracking.getByText(/Job IDs: 7101/);
      await expect(ids).toBeHidden();
      await screenshot(page, info, 'active-source-' + device);
      const summary = tracking.locator('summary');
      await summary.focus();
      await summary.press('Enter');
      await expect(ids).toBeVisible();
      await expect(summary).toBeFocused();
      for (const job of fixture.jobs.values()) job.polls = 0;
      await expect(dialog.getByRole('heading', { name: '2 saved to Media' })).toBeVisible({ timeout: 30_000 });
    });

    test('preview adds the selected playlist item and companion link exactly once', async ({ page }, info) => {
      const fixture = await installBrowserFixture(page, { webUrl, playlistPreflight: true, processingPolls: 0 });
      const playlist = 'https://www.youtube.com/playlist?list=PLestuary';
      const video = 'https://www.youtube.com/watch?v=b1234567890';
      await page.route('**/api/v1/media/playlists/preflight', route =>
        route.request().method() === 'POST' ? route.fulfill({
          headers: crossOrigin, json: {
            playlist_title: 'Estuary field recordings', playlist_id: 'PLestuary', item_count: 2,
            selected_count: 1, duplicate_count: 1, items: [
              { ordinal: 1, title: 'Earlier field recording', source_url: 'https://www.youtube.com/watch?v=a1234567890', duplicate_status: 'duplicate_existing', selected: false },
              { ordinal: 2, title: 'North bank field recording', source_url: video, duplicate_status: 'new', selected: true },
            ],
          },
        }) : route.fallback());
      const dialog = await openIngest(page);
      await fillLinks(dialog, [playlist, saved]);
      await dialog.getByRole('button', { name: 'Preview', exact: true }).click();
      await expect(dialog.getByRole('radio', { name: 'Skip duplicates' })).toBeChecked();
      await expect(dialog.getByText(/left out/i)).toBeVisible();
      await expect(dialog.getByRole('button', { name: /add urls/i })).toHaveCount(0);
      const add = dialog.getByRole('button', { name: 'Add 1 selected and 1 other link' });
      await expect(add).toBeEnabled();
      await add.scrollIntoViewIfNeeded();
      await screenshot(page, info, 'playlist-selection-' + device);
      await add.click();
      await expect(dialog.getByRole('textbox', { name: /URLs input/i })).toHaveValue('');
      await quickPreset(dialog);
      await dialog.getByRole('button', { name: 'Save 2 sources' }).click();
      await expect(dialog.getByRole('heading', { name: '2 saved to Media' })).toBeVisible({ timeout: 30_000 });
      expect([...fixture.attempts].sort()).toEqual([[saved, 1], [video, 1]].sort());
    });
  });
}
