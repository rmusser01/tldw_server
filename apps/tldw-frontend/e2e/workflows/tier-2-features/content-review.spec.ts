/**
 * UAT389 / B-09 bounded draft acceptance against an explicitly owned runtime.
 * Auth is seeded by WorkflowFixtures; this is not fresh-login certification.
 * Requires TLDW_LIVE_TIER_UAT=1 (no shared API stubs), real ingestion and text model.
 * Does not cover B-09 saved-source reanalysis/fallback or partial batch recovery.
 */
import { test, expect, assertNoCriticalErrors } from '../../utils/fixtures';
import { ContentReviewPage } from '../../utils/page-objects/ContentReviewPage';
import {
  ingestReviewDrafts,
  openReviewDraft,
  readReviewDrafts,
  assertSavedDraft,
  assertReviewDiff,
  successfulReviewResponse,
  assertCommittedMedia,
} from '../../utils/content-review-uat389';

const original = 'Speaker 1: The sample contains ORBIT-742.';
const edited = 'Speaker 1: The reviewed sample contains ORBIT-742.';

test.describe('Content Review — owned draft acceptance', () => {
  test.setTimeout(180_000);

  test.beforeEach(async ({ serverInfo }) => {
    expect(
      process.env.TLDW_LIVE_TIER_UAT,
      'UAT389 requires TLDW_LIVE_TIER_UAT=1 so auth setup does not stub app APIs'
    ).toBe('1');
    expect(serverInfo.available, 'UAT389 prerequisite: owned API is unavailable').toBe(true);
  });

  test('empty draft workspace opens the ingestion entry action', async ({
    authedPage,
    diagnostics,
  }) => {
    const review = new ContentReviewPage(authedPage);
    await review.goto();
    await review.assertPageReady();
    await expect(review.emptyState).toBeVisible();
    await expect(review.titleInput).toHaveCount(0);
    await review.openQuickIngestButton.click();
    await expect(authedPage.getByRole('dialog', { name: /quick ingest/i })).toBeVisible();
    await assertNoCriticalErrors(diagnostics);
  });

  test('saves exact edits and revisions, reopens Diff, and resets original content', async ({
    authedPage,
    diagnostics,
  }) => {
    const [draft] = await ingestReviewDrafts(authedPage, [original]);
    const review = new ContentReviewPage(authedPage);
    const changed = { ...draft, title: `${draft.title} reviewed`, content: edited };
    await review.titleInput.fill(changed.title);
    await review.contentTextarea.fill(changed.content);
    await review.saveDraftButton.click();
    const saved = await assertSavedDraft(authedPage, changed, 'Manual save');
    await authedPage.reload();
    await openReviewDraft(authedPage, saved);
    expect(
      (await readReviewDrafts(authedPage)).find((item) => item.id === draft.id)?.revisions
    ).toEqual(saved.revisions);
    await assertReviewDiff(review, original, edited);
    await review.resetButton.click();
    await authedPage
      .getByRole('dialog', { name: 'Reset draft?' })
      .getByRole('button', { name: 'Reset', exact: true })
      .click();
    await expect(review.contentTextarea).toHaveValue(original);
    await review.saveDraftButton.click();
    await expect
      .poll(
        async () =>
          (await readReviewDrafts(authedPage)).find((item) => item.id === draft.id)?.content
      )
      .toBe(original);
    await authedPage.reload();
    await openReviewDraft(authedPage, { ...saved, content: original });
    await assertNoCriticalErrors(diagnostics);
  });

  test('AI fix persists an inspectable proposal without changing another draft', async ({
    authedPage,
    serverInfo,
    diagnostics,
  }) => {
    expect(
      serverInfo.models?.length,
      'UAT389 prerequisite: a runnable text model is required'
    ).toBeGreaterThan(0);
    const [draft, other] = await ingestReviewDrafts(authedPage, [
      'Speaker 1: teh sample contains ORBIT-742.',
      'Speaker 2: Keep SATURN-518 unchanged.',
    ]);
    const review = new ContentReviewPage(authedPage);
    await expect(review.aiFixButton).toBeEnabled();
    const responsePromise = authedPage.waitForResponse(
      (response) =>
        response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/v1/chat/completions'
    );
    await review.aiFixButton.click();
    await authedPage
      .getByRole('dialog', { name: 'Send draft to server?' })
      .getByRole('button', { name: 'Continue', exact: true })
      .click();
    const response = await responsePromise;
    const body = await successfulReviewResponse(response);
    expect(response.request().postDataJSON()).toMatchObject({
      stream: false,
      messages: expect.arrayContaining([
        {
          role: 'user',
          content: `Correct the transcript below.\n\n<<<CONTENT>>>\n${draft.content}\n<<<END>>>`,
        },
      ]),
    });
    expect(body.choices[0].message.content.trim()).not.toBe('');
    await expect(review.contentTextarea).toHaveValue(body.choices[0].message.content.trim());
    await expect(review.contentTextarea).not.toHaveValue(draft.content);
    const proposal = await review.contentTextarea.inputValue();
    expect(proposal).toContain('ORBIT-742');
    expect(proposal).not.toMatch(/\bteh\b/i);
    const saved = await assertSavedDraft(
      authedPage,
      { ...draft, content: proposal },
      'AI corrections'
    );
    await assertReviewDiff(review, draft.content, proposal);
    await authedPage.reload();
    await openReviewDraft(authedPage, saved);
    expect((await readReviewDrafts(authedPage)).find((item) => item.id === other.id)).toEqual(
      other
    );
    await assertNoCriticalErrors(diagnostics);
  });

  test('commits reviewed content to one canonical source and reloads its version', async ({
    authedPage,
    diagnostics,
  }, testInfo) => {
    const [draft] = await ingestReviewDrafts(authedPage, [original]);
    const review = new ContentReviewPage(authedPage);
    const changed = { ...draft, title: `${draft.title} committed`, content: edited };
    await review.titleInput.fill(changed.title);
    await review.contentTextarea.fill(edited);
    await review.saveDraftButton.click();
    await assertSavedDraft(authedPage, changed, 'Manual save');
    await assertReviewDiff(review, original, edited);
    const adds: string[] = [];
    authedPage.on('request', (request) => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/v1/media/add')
        adds.push(request.url());
    });
    const addedPromise = authedPage.waitForResponse(
      (response) =>
        response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/v1/media/add'
    );
    const updatedPromise = authedPage.waitForResponse(
      (response) =>
        response.request().method() === 'PUT' &&
        /\/api\/v1\/media\/\d+$/.test(new URL(response.url()).pathname)
    );
    // Attach both waiters before clicking so a missing request always fails.
    const commitResponses = Promise.all([addedPromise, updatedPromise]);
    await expect(review.commitButton).toBeEnabled();
    await review.commitButton.click();
    const [addResponse, updateResponse] = await commitResponses;
    const added = await successfulReviewResponse(addResponse);
    expect(added.results).toHaveLength(1);
    expect(added.results[0].status).toBe('Success');
    const mediaId = added.results[0].db_id;
    expect(Number.isInteger(mediaId) && mediaId > 0).toBe(true);
    expect(new URL(updateResponse.url()).pathname).toBe(`/api/v1/media/${mediaId}`);
    expect(updateResponse.request().postDataJSON()).toMatchObject({
      title: changed.title,
      content: edited,
    });
    const updated = await successfulReviewResponse(updateResponse);
    expect(updated.media_id).toBe(mediaId);
    expect(Number.isInteger(updated.new_version) && updated.new_version > 1).toBe(true);
    await expect(review.committedTag.first()).toHaveText('1 committed');
    await expect(review.commitButton).toBeDisabled();
    const canonical = await assertCommittedMedia(mediaId, updated.new_version, changed);
    await authedPage.reload();
    await openReviewDraft(authedPage, changed);
    await expect(review.commitButton).toBeDisabled();
    expect(await assertCommittedMedia(mediaId, updated.new_version, changed)).toEqual(canonical);
    expect(adds).toHaveLength(1);
    await review.clearDraftsButton.click();
    await authedPage
      .getByRole('dialog', { name: 'Clear all drafts?' })
      .getByRole('button', { name: 'Clear drafts', exact: true })
      .click();
    await expect(review.emptyState).toBeVisible();
    expect(await assertCommittedMedia(mediaId, updated.new_version, changed)).toEqual(canonical);
    await testInfo.attach('uat389-committed-identity.json', {
      body: JSON.stringify({
        draftId: draft.id,
        batchId: draft.batchId,
        mediaId,
        version: updated.new_version,
        title: changed.title,
        content: changed.content,
      }),
      contentType: 'application/json',
    });
    await assertNoCriticalErrors(diagnostics);
  });
});
