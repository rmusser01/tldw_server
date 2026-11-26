import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { readReviewDrafts } from './utils/content-review-uat389';
import { installBrowserFixture } from './utils/media-ux-fixture';
import { ContentReviewPage } from './utils/page-objects/ContentReviewPage';

const webUrl = new URL(process.env.TLDW_WEB_URL || 'http://localhost:8080').origin;
const source = 'https://example.org/river-survey';
test.setTimeout(120_000);

async function prepareReviewDraft(page: Page) {
  await page.goto(webUrl + '/media', { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Add media', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Quick Ingest', exact: true });
  await dialog.getByRole('button', { name: 'Links', exact: true }).click();
  await dialog.getByRole('textbox', { name: /URLs input/i }).fill(source);
  await dialog.getByRole('button', { name: /add urls/i }).click();
  await dialog.getByRole('button', { name: 'Change settings' }).click();
  await dialog.getByLabel('Processing preset').selectOption('quick');
  await dialog.getByLabel('Destination', { exact: true }).selectOption('draft');
  return dialog;
}

/** Keep Dexie's first draft write uncommitted until the test aborts that transaction. */
async function holdFirstDraftWrite(page: Page) {
  return page.evaluateHandle(() => {
    const writing = new Promise<IDBTransaction>(resolve => {
      const put = IDBObjectStore.prototype.put;
      IDBObjectStore.prototype.put = function (...args) {
        const request = put.apply(this, args);
        if (this.name === 'contentDrafts') {
          IDBObjectStore.prototype.put = put;
          // IndexedDB automatically commits when there are no more pending requests.
          const keepActive = () => { this.count().onsuccess = keepActive; };
          keepActive();
          resolve(this.transaction);
        }
        return request;
      };
    });
    return async () => { (await writing).abort(); };
  });
}

async function capture(page: Page, info: TestInfo, name: string) {
  const path = info.outputPath(name + '.png');
  await page.screenshot({ path, animations: 'disabled' });
  await info.attach(name, { path, contentType: 'image/png' });
}

for (const [device, width, height] of [['desktop', 1280, 800], ['mobile', 390, 844]] as const) {
  test(`recovers an unsaved review draft without repeating extraction on ${device}`, async ({ page, context }, info) => {
    await page.setViewportSize({ width, height });
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: webUrl });
    const fixture = await installBrowserFixture(page, { webUrl, count: 2 });
    const dialog = await prepareReviewDraft(page);
    const abortWrite = await holdFirstDraftWrite(page);
    await dialog.getByRole('button', { name: 'Create 1 review draft' }).click();
    await expect(dialog.getByRole('heading', { name: 'Creating review drafts…' })).toBeVisible();
    const open = dialog.getByRole('button', { name: 'Open drafts', exact: true });
    await expect(open).toHaveCount(0);
    await abortWrite.evaluate(abort => abort());
    await expect(dialog.getByRole('heading', { name: 'Review drafts couldn’t be saved' })).toBeVisible();
    await expect(open).toHaveCount(0);
    expect(await readReviewDrafts(page)).toHaveLength(0);
    const retry = dialog.getByRole('button', { name: 'Retry saving drafts', exact: true });
    await expect(retry).toBeInViewport({ ratio: 1 });
    await capture(page, info, 'draft-save-failed-' + device);

    const view = dialog.getByRole('button', { name: /^View text for / });
    await view.click();
    const text = await dialog.getByRole('region', { name: 'Extracted text' }).locator('pre').innerText();
    expect(text).toContain('Three teams returned to the river');
    await dialog.getByRole('button', { name: 'Copy text', exact: true }).click();
    await expect(dialog.getByText('Text copied.', { exact: true })).toBeVisible();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(text);
    await dialog.getByRole('button', { name: 'Back to results' }).click();
    await expect(view).toBeFocused();

    await retry.click();
    await expect(dialog.getByRole('heading', { name: '1 draft ready for review' })).toBeVisible();
    await expect(open).toBeInViewport({ ratio: 1 });
    await capture(page, info, 'draft-saved-' + device);
    const drafts = await readReviewDrafts(page);
    expect(drafts).toHaveLength(1);
    expect(drafts[0].content).toBe(text);
    await open.click();
    await expect(page).toHaveURL(
      url => url.pathname === '/content-review' && url.searchParams.get('batch') === drafts[0].batchId,
      { timeout: 60_000 }
    );
    await expect(new ContentReviewPage(page).contentTextarea).toHaveValue(text);
    expect(fixture.attempts.get(source)).toBe(1);
    expect(fixture.jobs.size).toBe(0);
    expect(fixture.items).toHaveLength(2);
  });
}

test('an empty extraction offers no drafts to open or save again', async ({ page }) => {
  await installBrowserFixture(page, { webUrl });
  await page.route('**/api/v1/media/process-web-scraping', route =>
    route.request().method() === 'POST'
      ? route.fulfill({
          headers: { 'access-control-allow-origin': '*' },
          json: { status: 'ephemeral-ok', total_articles: 0, articles: [] },
        })
      : route.fallback()
  );
  const dialog = await prepareReviewDraft(page);
  await dialog.getByRole('button', { name: 'Create 1 review draft' }).click();
  await expect(dialog.getByRole('heading', { name: 'No review drafts created' })).toBeVisible();
  await expect(dialog.getByRole('button', { name: /Open drafts|Retry saving drafts/i })).toHaveCount(0);
  expect(await readReviewDrafts(page)).toHaveLength(0);
});
