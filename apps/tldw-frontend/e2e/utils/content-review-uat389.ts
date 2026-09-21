/** UAT389: real ingestion and read-only persistence oracles; no API stubs. */
import { randomUUID } from 'node:crypto';
import { expect, type Page, type Response } from '@playwright/test';
import type { ContentDraft } from '../../../packages/ui/src/db/dexie/types';
import { ContentReviewPage } from './page-objects/ContentReviewPage';
import { fetchWithApiKey, TEST_CONFIG } from './helpers';

export async function readReviewDrafts(page: Page): Promise<ContentDraft[]> {
  return page.evaluate(
    () =>
      new Promise((resolve, reject) => {
        const request = indexedDB.open('PageAssistDatabase');
        request.onupgradeneeded = () => {
          request.transaction?.abort();
          reject(new Error('UAT389 prerequisite: application draft database does not exist'));
        };
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const database = request.result;
          if (!database.objectStoreNames.contains('contentDrafts')) {
            database.close();
            reject(new Error('UAT389 prerequisite: contentDrafts store missing'));
            return;
          }
          const transaction = database.transaction('contentDrafts', 'readonly');
          const drafts = transaction.objectStore('contentDrafts').getAll();
          drafts.onsuccess = () => resolve(drafts.result);
          drafts.onerror = () => reject(drafts.error);
          transaction.oncomplete = () => database.close();
          transaction.onabort = () => {
            database.close();
            reject(transaction.error);
          };
        };
      })
  );
}

export async function openReviewDraft(page: Page, draft: ContentDraft): Promise<ContentReviewPage> {
  await page.goto(
    `/content-review?batch=${encodeURIComponent(draft.batchId)}&draft=${encodeURIComponent(draft.id)}`
  );
  const review = new ContentReviewPage(page);
  await review.assertPageReady();
  await expect(review.titleInput).toHaveValue(draft.title);
  await expect(review.contentTextarea).toHaveValue(draft.content);
  await expect(page).toHaveURL(new RegExp(`draft=${draft.id}(?:&|$)`));
  return review;
}

/** Generate the fixture through the supported file-upload/review option. */
export async function ingestReviewDrafts(page: Page, contents: string[]): Promise<ContentDraft[]> {
  const review = new ContentReviewPage(page);
  await review.goto();
  await review.assertPageReady();
  await expect(review.emptyState).toBeVisible();
  const files = contents.map((content, index) => ({
    name: `uat389-${randomUUID()}-${index}.txt`,
    mimeType: 'text/plain',
    buffer: Buffer.from(content),
  }));
  await review.openQuickIngestButton.click();
  const dialog = page.getByRole('dialog', { name: /quick ingest/i });
  await expect(dialog).toBeVisible();
  await dialog
    .locator('[data-testid="qi-file-input"], input[type="file"]')
    .first()
    .setInputFiles(files);
  await dialog
    .getByRole('button', { name: new RegExp(`configure ${files.length} items`, 'i') })
    .click();
  for (const option of ['analysis', 'chunking']) {
    const toggle = dialog.getByRole('switch', {
      name: new RegExp(`^Ingestion options\\s*[–-]\\s*${option}$`, 'i'),
    });
    await expect(toggle).toBeVisible();
    await toggle.setChecked(false);
  }
  await dialog.getByRole('button', { name: /^advanced options$/i }).click();
  await dialog.getByRole('switch', { name: 'Review before saving', exact: true }).setChecked(true);
  await dialog.getByRole('button', { name: /^next$/i }).click();
  await dialog.getByRole('button', { name: /start processing/i }).click();

  // UAT396: require the actual Quick Ingest handoff before inspecting drafts.
  await expect(page).toHaveURL(
    (url) => url.pathname === '/content-review' && Boolean(url.searchParams.get('batch')),
    { timeout: 120_000 }
  );
  await expect
    .poll(
      async () => {
        const drafts = await readReviewDrafts(page);
        return drafts.filter((draft) => files.some((file) => file.name === draft.source.fileName))
          .length;
      },
      {
        timeout: 120_000,
        message: 'UAT389 prerequisite failed: real Quick Ingest did not produce every owned draft',
      }
    )
    .toBe(files.length);
  const allDrafts = await readReviewDrafts(page);
  const drafts = files.map((file, index) => {
    const matches = allDrafts.filter((draft) => draft.source.fileName === file.name);
    expect(matches).toHaveLength(1);
    const draft = matches[0];
    expect(draft.originalContent).toBe(contents[index]);
    expect(draft.content).toBe(contents[index]);
    expect(draft.title.trim()).not.toBe('');
    expect(draft.sourceAssetId).toBeTruthy();
    expect(draft.revisions).toEqual([]);
    return draft;
  });
  expect(new Set(drafts.map((draft) => draft.batchId)).size).toBe(1);
  expect(new Set(drafts.map((draft) => draft.id)).size).toBe(files.length);
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get('batch') === drafts[0].batchId &&
      drafts.some((draft) => draft.id === url.searchParams.get('draft'))
  );
  const selected = drafts.find(
    (draft) => draft.id === new URL(page.url()).searchParams.get('draft')
  )!;
  await review.assertPageReady();
  await expect(review.titleInput).toHaveValue(selected.title);
  await expect(review.contentTextarea).toHaveValue(selected.content);
  await openReviewDraft(page, drafts[0]);
  return drafts;
}

export async function assertSavedDraft(
  page: Page,
  expected: ContentDraft,
  revisionLabel: string
): Promise<ContentDraft> {
  await expect
    .poll(
      async () => {
        const draft = (await readReviewDrafts(page)).find((item) => item.id === expected.id);
        return (
          draft && {
            title: draft.title,
            content: draft.content,
            originalContent: draft.originalContent,
            revision: draft.revisions[0]?.content,
            label: draft.revisions[0]?.changeDescription,
          }
        );
      },
      { message: 'UAT389: exact local content and revision were not persisted' }
    )
    .toEqual({
      title: expected.title,
      content: expected.content,
      originalContent: expected.originalContent,
      revision: expected.content,
      label: revisionLabel,
    });
  const saved = (await readReviewDrafts(page)).find((item) => item.id === expected.id)!;
  expect(saved.revisions[0].id).toBeTruthy();
  expect(saved.revisions[0].timestamp).toBeGreaterThan(expected.createdAt);
  return saved;
}

export async function assertReviewDiff(
  review: ContentReviewPage,
  original: string,
  edited: string
): Promise<void> {
  await review.diffViewButton.click();
  await expect(review.diffModal).toBeVisible();
  const region = review.diffModal.getByRole('region', { name: /diff content/i });
  // Separate removed and added rows: a modal that repeats the same side cannot pass.
  await expect(region.locator('.text-danger .whitespace-pre-wrap')).toHaveText(
    original.split('\n')
  );
  await expect(region.locator('.text-success .whitespace-pre-wrap')).toHaveText(edited.split('\n'));
  await review.diffModal.getByRole('button', { name: /close/i }).click();
  await expect(review.diffModal).toBeHidden();
}

export async function successfulReviewResponse(response: Response) {
  expect(
    response.ok(),
    `${response.request().method()} ${new URL(response.url()).pathname} returned ${response.status()}`
  ).toBe(true);
  return response.json();
}

export async function readCommittedMedia(mediaId: number) {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/${mediaId}?include_version_content=true`
  );
  expect(response.ok, `Canonical media ${mediaId} read failed: ${response.status}`).toBe(true);
  return response.json();
}

/** The server schema is MediaDetailResponse and VersionDetailResponse. */
export async function assertCommittedMedia(mediaId: number, version: number, draft: ContentDraft) {
  const media = await readCommittedMedia(mediaId);
  expect(media.media_id).toBe(mediaId);
  expect(media.source.title).toBe(draft.title);
  expect(media.content.text).toBe(draft.content);
  const savedVersion = media.versions.find(
    (item: { version_number: number }) => item.version_number === version
  );
  expect(savedVersion).toMatchObject({
    media_id: mediaId,
    version_number: version,
    content: draft.content,
  });
  expect(
    media.versions.some(
      (item: { version_number: number; content: string }) =>
        item.version_number < version && item.content === draft.originalContent
    )
  ).toBe(true);
  return {
    mediaId: media.media_id,
    title: media.source.title,
    content: media.content.text,
    versions: media.versions
      .map((item: { media_id: number; version_number: number; content: string }) => ({
        mediaId: item.media_id,
        version: item.version_number,
        content: item.content,
      }))
      .sort((a: { version: number }, b: { version: number }) => a.version - b.version),
  };
}
