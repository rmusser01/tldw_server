import { test, expect, type Locator, type Page, type TestInfo } from '@playwright/test';
import { installBrowserFixture } from './utils/media-ux-fixture';

const webUrl = new URL(process.env.TLDW_WEB_URL || 'http://localhost:8080').origin;
test.setTimeout(120_000);
test.use({ actionTimeout: 20_000 });

async function openIngest(page: Page) {
  await page.goto(webUrl + '/media', { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Add media', exact: true }).click();
  return page.getByRole('dialog', { name: 'Quick Ingest', exact: true });
}

async function addLinks(dialog: Locator, links: string[]) {
  await dialog.getByRole('button', { name: 'Links', exact: true }).click();
  await dialog.getByRole('textbox', { name: /URLs input/i }).fill(links.join('\n'));
  await dialog.getByRole('button', { name: /add urls/i }).click();
}

async function useQuickPreset(dialog: Locator) {
  await dialog.getByRole('button', { name: 'Change settings', exact: true }).click();
  await dialog.getByLabel('Processing preset').selectOption('quick');
}

async function capture(page: Page, info: TestInfo, name: string) {
  const path = info.outputPath(name + '.png');
  await page.screenshot({ path, animations: 'disabled' });
  await info.attach(name, { path, contentType: 'image/png' });
}

for (const [device, width, height] of [['desktop', 1280, 800], ['mobile', 390, 844]] as const) {
  test(`preserves the draft through playlist replacement review and Undo on ${device}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height });
    const fixture = await installBrowserFixture(page, { webUrl, playlistPreflight: true, processingPolls: 0 });
    const items = [
      { ordinal: 1, title: 'Field recording techniques', source_url: 'https://www.youtube.com/watch?v=a1234567890', duplicate_status: 'duplicate_existing', selected: false },
      { ordinal: 2, title: 'Interview notes in practice', source_url: 'https://www.youtube.com/watch?v=b1234567890', duplicate_status: 'new', selected: true },
    ];
    await page.route('**/api/v1/media/playlists/preflight', route => route.fulfill({
      headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': 'POST,OPTIONS' },
      json: { playlist_title: 'Oral history workshop', playlist_id: 'PLfieldnotes', item_count: 2, selected_count: 1, duplicate_count: 1, items },
    }));
    const dialog = await openIngest(page);
    await dialog.getByRole('button', { name: 'Links', exact: true }).click();
    const links = dialog.getByRole('textbox', { name: /URLs input/i });
    await links.fill('https://www.youtube.com/playlist?list=PLfieldnotes');
    await dialog.getByRole('button', { name: 'Preview', exact: true }).click();
    await dialog.getByRole('radio', { name: 'Overwrite', exact: true }).check();
    await dialog.getByRole('button', { name: 'Add 2 selected', exact: true }).click();
    await useQuickPreset(dialog);
    await dialog.getByRole('button', { name: 'Hide settings', exact: true }).click();
    await links.fill('https://example.org/interview-index');
    await dialog.getByRole('button', { name: 'Text', exact: true }).click();
    const passage = dialog.getByRole('textbox', { name: 'Pasted text input' });
    await passage.fill('Follow up with the archivist about the interview dates.');
    const editor = dialog.locator('.qi-web-scroll:visible');
    const scrollBeforeReview = await editor.evaluate(el => el.scrollTop);
    const review = dialog.getByRole('button', { name: 'Review replacements', exact: true });
    await review.click();
    await expect(dialog.getByRole('heading', { name: 'Review replacements' })).toBeFocused();
    await expect(dialog.getByText('2 sources will run', { exact: true })).toBeVisible();
    await expect(dialog.getByText('1 existing Media item may be replaced.', { exact: true })).toBeVisible();
    const affected = dialog.getByRole('listitem');
    await expect(affected).toHaveCount(1);
    await expect(affected).toContainText(items[0].title);
    await expect(affected).toContainText(items[0].source_url);
    const back = dialog.getByRole('button', { name: 'Back to sources' });
    const confirm = dialog.getByRole('button', { name: 'Confirm and save 2 sources' });
    await expect(back).toBeInViewport({ ratio: 1 });
    await expect(confirm).toBeInViewport({ ratio: 1 });
    if (device === 'mobile') {
      for (const button of [back, confirm]) expect((await button.boundingBox())!.width).toBeGreaterThan(width * 0.85);
    }
    await capture(page, info, 'replacement-review-' + device);
    await back.focus();
    await back.press('Enter');
    await expect(review).toBeFocused();
    await expect(passage).toHaveValue('Follow up with the archivist about the interview dates.');
    expect(await editor.evaluate(el => el.scrollTop)).toBe(scrollBeforeReview);
    await dialog.getByRole('button', { name: 'Clear all', exact: true }).click();
    const undo = dialog.getByRole('button', { name: 'Undo', exact: true });
    await expect(undo).toBeFocused();
    await expect(dialog.getByText('2 sources cleared.', { exact: true })).toBeVisible();
    await capture(page, info, 'undo-clear-' + device);
    await undo.click();
    await expect(dialog.getByRole('button', { name: 'Clear all' })).toBeFocused();
    await dialog.getByRole('button', { name: 'Links', exact: true }).click();
    await expect(links).toHaveValue('https://example.org/interview-index');
    expect(fixture.jobs.size).toBe(0);
    await review.click();
    await dialog.getByRole('button', { name: 'Confirm and save 2 sources' }).click();
    await expect(dialog.getByRole('heading', { name: '2 saved to Media' })).toBeVisible();
    expect([...fixture.jobs.values()].map(job => job.source_url)).toEqual(items.map(item => item.source_url));
    const submissions = fixture.requests.filter(request => request.method === 'POST' && request.path === '/api/v1/media/ingest/jobs');
    expect(submissions.map(request => request.body.overwrite_existing)).toEqual([true, false]);
  });
}

test('Undo keeps pasted sources added before and after Clear all ready to run', async ({ page }) => {
  await installBrowserFixture(page, { webUrl });
  const dialog = await openIngest(page);
  await dialog.getByRole('button', { name: 'Text', exact: true }).click();
  const paste = async (value: string) => {
    await dialog.getByRole('textbox', { name: 'Pasted text input' }).fill(value);
    await dialog.getByRole('button', { name: /Add pasted text to queue/ }).click();
  };
  await paste('Map');
  await paste('Log');
  await dialog.getByRole('button', { name: 'Clear all' }).click();
  await paste('Key');
  await dialog.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(dialog.getByRole('status', { name: 'Source counts' })).toHaveText('3 ready');
});

test('processing confirmations keep every action in reach on a narrow phone', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 700 });
  await installBrowserFixture(page, { webUrl, processingPolls: 100 });
  const dialog = await openIngest(page);
  await dialog.getByTestId('qi-file-input').setInputFiles({
    name: 'interview.txt', mimeType: 'text/plain', buffer: Buffer.from('Field interview'),
  });
  await useQuickPreset(dialog);
  await dialog.getByRole('button', { name: 'Save 1 source' }).click();
  await expect(dialog.getByText('40%', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  const confirm = page.getByRole('dialog', { name: 'Leave this import?' });
  const stay = confirm.getByRole('button', { name: 'Stay here' });
  for (const button of await confirm.getByRole('button').all()) {
    await expect(button).toBeInViewport({ ratio: 1 });
  }
  await expect(stay).toBeFocused();
  await stay.click();
  await dialog.getByRole('button', { name: 'Cancel All', exact: true }).click();
  const stop = page.getByRole('dialog', { name: 'Stop the remaining import?' });
  await expect(stop.getByText('1 source is still active.')).toBeVisible();
  await stop.getByRole('button', { name: 'Keep importing' }).click();
  await expect(dialog.getByText('40%', { exact: true })).toBeVisible();
});

for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
  test(`reads and copies ephemeral web content at ${viewport.width}px`, async ({ page, context }, info) => {
    await page.setViewportSize(viewport);
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: webUrl });
    const fixture = await installBrowserFixture(page, { webUrl, count: 2 });
    const dialog = await openIngest(page);
    await addLinks(dialog, ['https://example.org/river-survey']);
    await useQuickPreset(dialog);
    await dialog.getByLabel('Destination', { exact: true }).selectOption('extract');
    await dialog.getByRole('button', { name: 'Extract 1 source' }).click();
    const view = dialog.getByRole('button', { name: 'View extracted text', exact: true });
    await view.click();
    const preview = dialog.getByRole('region', { name: 'Extracted text', exact: true });
    const text = await preview.locator('pre').textContent();
    expect(text).toContain('Three teams returned to the river');
    const copy = dialog.getByRole('button', { name: 'Copy text', exact: true });
    await expect(copy).toBeInViewport({ ratio: 1 });
    await capture(page, info, 'extraction-' + viewport.width);
    await copy.click();
    await expect(dialog.getByText('Text copied.', { exact: true })).toBeVisible();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(text);
    await dialog.getByRole('button', { name: 'Back to results' }).click();
    await expect(view).toBeFocused();
    expect(fixture.items).toHaveLength(2);
  });
}

test('restores a file and a pasted source on their original phone draft rows', async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installBrowserFixture(page, { webUrl });
  const dialog = await openIngest(page);
  const file = { name: 'Interview notes.txt', mimeType: 'text/plain', buffer: Buffer.from('Source notes') };
  await dialog.getByTestId('qi-file-input').setInputFiles(file);
  await dialog.getByRole('button', { name: 'Text', exact: true }).click();
  await dialog.getByRole('textbox', { name: 'Pasted text input' }).fill('Observation from the site visit.');
  await dialog.getByRole('button', { name: /Add pasted text to queue/i }).click();
  await useQuickPreset(dialog);
  await expect(dialog.getByRole('button', { name: 'Save 2 sources' })).toBeEnabled();
  await page.reload({ waitUntil: 'domcontentloaded' });
  const trigger = page.getByRole('button', { name: 'Add media', exact: true });
  await expect(dialog.or(trigger).first()).toBeVisible();
  if (!await dialog.isVisible()) await trigger.click();
  await expect(dialog.getByRole('status', { name: 'Source counts' })).toContainText('0 ready');
  const picker = dialog.getByLabel('Choose file for Interview notes.txt', { exact: true });
  await dialog.getByRole('button', { name: 'Paste again', exact: true }).scrollIntoViewIfNeeded();
  await capture(page, info, 'source-repair-mobile');
  await picker.setInputFiles(file);
  await dialog.getByRole('button', { name: 'Paste again', exact: true }).click();
  const editor = dialog.getByRole('region', { name: 'Queued sources' }).getByRole('textbox');
  await editor.fill('Observation from the site visit.');
  await dialog.getByRole('button', { name: 'Apply text', exact: true }).click();
  await expect(dialog.getByRole('button', { name: 'Save 2 sources' })).toBeEnabled();
});

test('keeps a mixed import visible and reconciles a rejected stop', async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const fixture = await installBrowserFixture(page, { webUrl, processingPolls: 100 });
  await page.route('**/api/v1/media/ingest/jobs/cancel**', route => {
    if (route.request().method() === 'OPTIONS') return route.fallback();
    return route.fulfill({
      status: 503, headers: { 'access-control-allow-origin': '*' },
      json: { detail: 'Cancellation is temporarily unavailable' },
    });
  });
  const dialog = await openIngest(page);
  await addLinks(dialog, ['https://example.org/fail-once']);
  await dialog.getByRole('button', { name: 'Files', exact: true }).click();
  await dialog.getByTestId('qi-file-input').setInputFiles({
    name: 'River survey.txt', mimeType: 'text/plain', buffer: Buffer.from('Field notebook'),
  });
  await useQuickPreset(dialog);
  await dialog.getByRole('button', { name: 'Save 2 sources' }).click();
  await expect(dialog.getByText('40%', { exact: true })).toBeVisible();
  await dialog.getByRole('button', { name: 'Minimize to Background' }).click();
  const widget = page.getByRole('status', { name: 'Ingest progress' });
  await expect(widget).toContainText('1 active · 1 failed');
  await capture(page, info, 'background-mobile');
  await widget.getByRole('button', { name: /open|expand/i }).first().click();
  await dialog.getByRole('button', { name: 'Cancel All', exact: true }).click();
  await page.getByRole('dialog', { name: 'Stop the remaining import?' })
    .getByRole('button', { name: 'Stop remaining sources' }).click();
  await expect(page.locator('.qi-processing-prompt').filter({ hasText: 'Stop remaining sources' })).toBeHidden();
  await expect(dialog.getByText(/Could not confirm the stop/)).toBeVisible();
  await capture(page, info, 'unconfirmed-stop-mobile');
  for (const job of fixture.jobs.values()) job.polls = 0;
  await expect(dialog.getByRole('heading', { name: '1 saved to Media' })).toBeVisible({ timeout: 30_000 });
  await expect(dialog.getByText('1 cancelled', { exact: true })).toHaveCount(0);
});

test('repairs a server-rejected link without repeating the saved source', async ({ page }, info) => {
  const fixture = await installBrowserFixture(page, { webUrl, count: 2 });
  const original = 'https://example.org/old-interview';
  const corrected = 'https://example.org/corrected-interview';
  const saved = 'https://example.org/field-notes';
  await page.route('**/api/v1/media/process-web-scraping', async route => {
    if (route.request().method() === 'POST' && route.request().postDataJSON()?.url_input === original) {
      await route.fulfill({
        status: 422, headers: { 'access-control-allow-origin': '*' }, json: { detail: 'Unsupported URL format' },
      });
    } else await route.fallback();
  });
  const dialog = await openIngest(page);
  await addLinks(dialog, [original, saved]);
  await useQuickPreset(dialog);
  await dialog.getByRole('button', { name: 'Save 2 sources' }).click();
  await expect(dialog.getByRole('heading', { name: '1 saved to Media' })).toBeVisible();
  await dialog.getByRole('button', { name: /Edit link/i }).click();
  const editor = dialog.getByRole('textbox', { name: 'Correct link' });
  await expect(editor).toBeFocused();
  await editor.fill(corrected);
  await capture(page, info, 'repair-link-desktop');
  await dialog.getByRole('button', { name: 'Apply link' }).click();
  await dialog.getByRole('button', { name: 'Save 1 source' }).click();
  await expect(dialog.getByRole('heading', { name: '2 saved to Media' })).toBeVisible();
  expect(fixture.attempts.get(saved)).toBe(1);
  expect(fixture.attempts.get(corrected)).toBe(1);
});

test('checks an interrupted job without uploading it again', async ({ page }) => {
  const fixture = await installBrowserFixture(page, { webUrl, processingPolls: 100 });
  let inaccessible = false;
  await page.route(/\/api\/v1\/media\/ingest\/jobs\/\d+$/, route => {
    if (!inaccessible || route.request().method() !== 'GET') return route.fallback();
    return route.fulfill({
      status: 404, headers: { 'access-control-allow-origin': '*' }, json: { detail: 'Job unavailable' },
    });
  });
  const dialog = await openIngest(page);
  await dialog.getByTestId('qi-file-input').setInputFiles({
    name: 'Survey notes.txt', mimeType: 'text/plain', buffer: Buffer.from('Survey observations.'),
  });
  await useQuickPreset(dialog);
  await dialog.getByRole('button', { name: 'Save 1 source' }).click();
  await expect(dialog.getByText('40%', { exact: true })).toBeVisible();
  inaccessible = true;

  const interrupted = dialog.getByRole('region', { name: 'Status unavailable' });
  await expect(interrupted).toContainText('Survey notes.txt');
  await expect(interrupted).toContainText('Last reported: Simulated processing outcome · 40%');
  inaccessible = false;
  for (const job of fixture.jobs.values()) job.polls = 0;
  await interrupted.getByRole('button', { name: 'Check status' }).click();
  await expect(dialog.getByRole('heading', { name: '1 saved to Media' })).toBeVisible();
  expect(fixture.jobs.size).toBe(1);
});

for (const width of [320, 390, 1280]) {
  test(`edits batch metadata within the modal at ${width}px`, async ({ page }, info) => {
    const phone = width < 600;
    await page.setViewportSize({ width, height: phone ? 844 : 800 });
    const fixture = await installBrowserFixture(page, { webUrl, processingPolls: 0 });
    const dialog = await openIngest(page);
    const urls = [
      'https://archive.example.org/workshop/shoreline.mp4',
      'https://archive.example.org/workshop/interviews.mp4' + (width === 320 ? '?record=' + 'b'.repeat(100) : ''),
    ];
    await addLinks(dialog, urls);
    await useQuickPreset(dialog);
    await dialog.getByRole('button', { name: 'Hide settings' }).click();
    const batch = dialog.getByRole('region', { name: 'Conference batch metadata' });
    await batch.getByLabel('Collection name').fill('Coastal oral histories');
    await batch.getByLabel('Conference name').fill('Field Recording Forum');
    await batch.getByLabel('Event year').fill('2026');
    await batch.getByLabel('Shared tags').fill('workshop');
    const first = batch.getByRole('group', { name: 'Source 1', exact: true });
    const second = batch.getByRole('group', { name: 'Source 2', exact: true });
    const secondToggle = second.getByRole('button');
    if (phone) {
      await expect(first.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
      await expect(second.getByRole('textbox')).toHaveCount(0);
      await secondToggle.focus();
      await secondToggle.press('Enter');
      await expect(secondToggle).toBeFocused();
      expect(await second.evaluate(el => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    }
    await second.getByLabel('Title override for item 2').fill('Interview techniques');
    await second.getByLabel('Speaker for item 2').fill('Sam Rivera');
    if (phone) await secondToggle.click();
    await second.getByRole('checkbox').uncheck();
    await expect(batch.getByText('1 selected')).toBeVisible();
    const title = first.getByLabel('Title override for item 1');
    const speaker = first.getByLabel('Speaker for item 1');
    const track = first.getByLabel('Track for item 1');
    const tags = first.getByLabel('Tags for item 1');
    await title.fill('Listening to tides');
    await speaker.fill('Maya Chen');
    await track.fill('Fieldwork');
    await tags.fill('coast, audio');
    await tags.press('Enter');

    const editorBox = (await batch.boundingBox())!;
    for (const control of [title, speaker, track, tags, first.getByRole('checkbox'), second.getByRole('checkbox')]) {
      const box = (await control.boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(Math.max(0, editorBox.x));
      expect(box.x + box.width).toBeLessThanOrEqual(Math.min(width, editorBox.x + editorBox.width) + 1);
    }
    const scrollAreas = dialog.locator('.qi-web-scroll:visible, .qi-metadata-editor, .qi-metadata-list, .qi-metadata-row');
    for (const area of await scrollAreas.all()) {
      expect(await area.evaluate(el => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    }
    if (phone) {
      expect((await track.boundingBox())!.y).toBeGreaterThan((await speaker.boundingBox())!.y);
      for (const field of [title, speaker, track, tags]) expect((await field.boundingBox())!.height).toBeGreaterThanOrEqual(44);
    } else {
      await expect(first.getByRole('button')).toHaveCount(0);
      expect((await track.boundingBox())!.y).toBe((await speaker.boundingBox())!.y);
    }
    const target = phone ? batch.locator('.qi-metadata-editor') : batch;
    await target.evaluate(el => el.scrollIntoView({ block: 'center' }));
    if (width !== 320) await capture(page, info, 'batch-metadata-' + (phone ? 'mobile' : 'desktop'));

    // A partially typed tag and keyboard focus must follow the source when columns reflow.
    if (!phone) {
      const secondTags = second.getByLabel('Tags for item 2');
      await secondTags.fill('archive, ');
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(secondTags).toBeFocused();
      await expect(secondTags).toHaveValue('archive, ');
      await expect(second.getByRole('checkbox')).not.toBeChecked();
      await expect(second.getByLabel('Speaker for item 2')).toHaveValue('Sam Rivera');
    }

    await dialog.getByRole('button', { name: 'Save 1 source' }).click();
    await expect(dialog.getByRole('heading', { name: '1 saved to Media' })).toBeVisible();
    const saved = fixture.requests.filter(r => r.path === '/api/v1/media/collections/21/items' && r.method === 'POST');
    expect(saved.map(r => r.body)).toEqual([expect.objectContaining({
      source_url: urls[0], title: 'Listening to tides', speaker: 'Maya Chen',
      track: 'Fieldwork', tags: ['workshop', 'coast', 'audio'],
    })]);
    expect([...fixture.jobs.values()].map(job => job.source_url)).toEqual([urls[0]]);
  });
}
