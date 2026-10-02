/**
 * A-08 -> A-10 -> A-11: real Note handoff, five generated/saved cards and five
 * Easy reviews. Run separately with a controlled backend provider and a live
 * provider. No application API is fulfilled or seeded by this journey.
 *
 * TLDW_LIVE_TIER_UAT=1 disables application API fulfillment in the shared fixture.
 * UAT_STUDY_PROVIDER / UAT_STUDY_MODEL optionally select the recorded provider.
 * Other A-10/A-11 controls (imports, practice, re-rate, lapse) remain separate.
 */
import { randomUUID } from 'node:crypto';
import { type Response } from '@playwright/test';
import type {
  Deck,
  Flashcard,
  FlashcardAnalyticsSummary,
  FlashcardListResponse,
  FlashcardNextReviewResponse,
  FlashcardReviewResponse,
  FlashcardReviewSessionSummary,
  FlashcardsGenerateResponse,
} from '@/services/flashcards';
import { test, expect } from '../../utils/fixtures';
import { NotesPage, FlashcardsPage } from '../../utils/page-objects';
import { expectApiCall } from '../../utils/api-assertions';
import { createNote } from '../../utils/journey-helpers';
import { fetchWithApiKey, TEST_CONFIG } from '../../utils/helpers';
import { assertBiologyCardSet, BIOLOGY_NOTE_CONTENT } from '../../utils/biology-card-oracle';

async function readApi<T>(path: string): Promise<T> {
  const response = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}${path}`);
  expect(response.ok, `Readback ${path}: HTTP ${response.status}`).toBe(true);
  return (await response.json()) as T;
}

test.describe('Notes -> Flashcards journey', () => {
  test('save five sourced generated cards and persist five Easy reviews', async ({
    authedPage: page,
    serverInfo,
  }, testInfo) => {
    test.setTimeout(300_000);
    expect(
      process.env.TLDW_LIVE_TIER_UAT,
      'Run with TLDW_LIVE_TIER_UAT=1 to disable application API stubs'
    ).toBe('1');
    expect(
      serverInfo.available,
      'Required backend is unavailable; this linked journey cannot pass'
    ).toBe(true);
    expect(serverInfo.models?.length, 'A configured text provider is required').toBeGreaterThan(0);

    const noteTitle = `uat-study-${randomUUID()}`;
    const deckName = `${noteTitle}-deck`;
    const notes = new NotesPage(page);
    const flashcards = new FlashcardsPage(page);
    const writes: Response[] = [];
    const evidence: Record<string, unknown> = { noteTitle, deckName };
    let noteId = '';
    let deck: Deck;
    let savedCards: Flashcard[] = [];
    let expectedDrafts: Array<{ front: string; back: string }> = [];
    const reviewedCards: FlashcardReviewResponse[] = [];
    const captureWrite = (response: Response) => {
      if (
        response.request().method() === 'POST' &&
        /\/api\/v1\/flashcards(?:\/generate|\/review)?$/.test(new URL(response.url()).pathname)
      ) {
        writes.push(response);
      }
    };
    page.on('response', captureWrite);

    try {
      await test.step('Create and reload the canonical F-BIOLOGY Note', async () => {
        const [{ response }] = await Promise.all([
          expectApiCall(page, { method: 'POST', url: /\/api\/v1\/notes\/?$/ }, 30_000),
          createNote(page, { title: noteTitle, content: BIOLOGY_NOTE_CONTENT }),
        ]);
        expect(response.ok()).toBe(true);
        const note = await response.json();
        expect(typeof note.id).toBe('string');
        expect(note.id).not.toBe('');
        noteId = note.id;
        evidence.noteId = noteId;
        expect(note).toMatchObject({ title: noteTitle, content: BIOLOGY_NOTE_CONTENT });
        await page.goto(`/notes?source_ref_id=${encodeURIComponent(noteId)}`);
        await notes.assertPageReady();
        await expect(notes.titleInput).toHaveValue(noteTitle);
        await notes.ensureMarkdownMode();
        await expect(notes.contentTextarea).toHaveValue(BIOLOGY_NOTE_CONTENT);
        expect(await readApi(`/api/v1/notes/${encodeURIComponent(noteId)}`)).toMatchObject({
          id: noteId,
          title: noteTitle,
          content: BIOLOGY_NOTE_CONTENT,
        });
      });

      await test.step('Generate exactly five distinct supported drafts through the Note handoff', async () => {
        await notes.overflowMenuButton.click();
        const handoff = page.getByRole('menuitem', { name: /^Generate flashcards$/i });
        await expect(handoff).toBeVisible();
        await handoff.click();
        await expect(page).toHaveURL(/\/flashcards\?/);
        await expect(flashcards.generateTextarea).toBeVisible();
        await expect(flashcards.generateTextarea).toHaveValue(BIOLOGY_NOTE_CONTENT);
        await expect(page.getByTestId('flashcards-generate-source-context')).toContainText(
          noteTitle
        );
        await page.getByTestId('flashcards-generate-count').fill('5');
        await expect(page.getByTestId('flashcards-generate-card-type')).toContainText('Basic');
        await page.getByTestId('flashcards-generate-deck').click();
        await flashcards.getActiveSelectOption('Create new deck', true).click();
        await page.getByTestId('flashcards-generate-new-deck-name').fill(deckName);
        if (process.env.UAT_STUDY_PROVIDER) {
          await page
            .getByTestId('flashcards-generate-provider')
            .fill(process.env.UAT_STUDY_PROVIDER);
        }
        if (process.env.UAT_STUDY_MODEL) {
          await page.getByTestId('flashcards-generate-model').fill(process.env.UAT_STUDY_MODEL);
        }
        await expect(flashcards.generateButton).toBeEnabled();
        const started = performance.now();
        const [{ request, response }] = await Promise.all([
          expectApiCall(page, { method: 'POST', url: /\/api\/v1\/flashcards\/generate$/ }, 180_000),
          (async () => {
            await flashcards.generateButton.click();
            await expect
              .poll(
                async () =>
                  (await flashcards.generateButton.getAttribute('class'))?.includes(
                    'ant-btn-loading'
                  ) || (await page.getByTestId('flashcards-generate-save-button').isVisible()),
                {
                  timeout: 2_000,
                  message: 'Generation must acknowledge progress or show drafts within 2 seconds',
                }
              )
              .toBe(true);
            evidence.generationAcknowledgedMs = performance.now() - started;
          })(),
        ]);
        evidence.generationCompletedMs = performance.now() - started;
        expect(request.postDataJSON()).toMatchObject({
          text: BIOLOGY_NOTE_CONTENT,
          num_cards: 5,
          card_type: 'basic',
        });
        expect(response.ok()).toBe(true);
        const generated = (await response.json()) as FlashcardsGenerateResponse;
        evidence.generation = { request: request.postDataJSON(), response: generated };
        expect(generated.count).toBe(5);
        assertBiologyCardSet(generated.flashcards);
        const draftCards = page
          .getByTestId('flashcards-create-generate-section')
          .locator('.ant-card')
          .filter({
            has: page.locator(':scope > .ant-card-head .ant-card-head-title', {
              hasText: /^Card \d+$/,
            }),
          });
        await expect(draftCards).toHaveCount(5);
        for (let index = 0; index < 5; index += 1) {
          await expect(draftCards.nth(index).locator('textarea').nth(0)).toHaveValue(
            generated.flashcards[index].front.trim()
          );
          await expect(draftCards.nth(index).locator('textarea').nth(1)).toHaveValue(
            generated.flashcards[index].back.trim()
          );
        }
        // A real draft edit must survive persistence. Keep the answer unchanged.
        const editedFront = `Recall from the source note: ${generated.flashcards[0].front.trim()}`;
        await draftCards.nth(0).locator('textarea').nth(0).fill(editedFront);
        evidence.editedFront = editedFront;
        expectedDrafts = generated.flashcards.map((card, index) => ({
          front: index === 0 ? editedFront : card.front.trim(),
          back: card.back.trim(),
        }));
      });

      await test.step('Save once; read back five canonical cards and their source links after reload', async () => {
        const save = page.getByTestId('flashcards-generate-save-button');
        await expect(save).toBeEnabled();
        const [{ response: deckResponse }] = await Promise.all([
          expectApiCall(page, { method: 'POST', url: /\/api\/v1\/flashcards\/decks$/ }, 30_000),
          save.click(),
        ]);
        expect(deckResponse.ok()).toBe(true);
        deck = (await deckResponse.json()) as Deck;
        expect(deck.id).toBeGreaterThan(0);
        expect(deck.name).toBe(deckName);
        evidence.deck = deck;
        await expect(page.getByTestId('flashcards-generate-save-status')).toContainText(
          'Saved 5 generated cards.'
        );
        await expect(save).toBeHidden();
        const saves = writes.filter((response) =>
          /\/api\/v1\/flashcards$/.test(new URL(response.url()).pathname)
        );
        expect(saves).toHaveLength(5);
        for (const response of saves) {
          expect(response.ok()).toBe(true);
          expect(response.request().postDataJSON()).toMatchObject({
            deck_id: deck.id,
            source_ref_type: 'note',
            source_ref_id: noteId,
          });
        }
        savedCards = (await Promise.all(saves.map((response) => response.json()))) as Flashcard[];
        expect(new Set(savedCards.map((card) => card.uuid)).size).toBe(5);
        expect(savedCards.map(({ front, back }) => ({ front, back }))).toEqual(expectedDrafts);
        evidence.savedCards = savedCards;
        await page.goto(`/flashcards?tab=manage&deck_id=${deck.id}`);
        await flashcards.assertPageReady();
        await flashcards.switchToTab('manage');
        await flashcards.selectManageDeckByName(deckName);
        const stored = await readApi<FlashcardListResponse>(
          `/api/v1/flashcards?deck_id=${deck.id}&due_status=all&limit=100`
        );
        expect(stored.total).toBe(5);
        expect(stored.items).toHaveLength(5);
        expect(stored.items.map((card) => card.uuid).sort()).toEqual(
          savedCards.map((card) => card.uuid).sort()
        );
        for (const card of savedCards) {
          expect(typeof card.uuid).toBe('string');
          expect(card.uuid).not.toBe('');
          const persisted = stored.items.find((item) => item.uuid === card.uuid);
          expect(persisted).toMatchObject({
            uuid: card.uuid,
            front: card.front,
            back: card.back,
            deck_id: deck.id,
            source_ref_type: 'note',
            source_ref_id: noteId,
            model_type: 'basic',
            reverse: false,
            is_cloze: false,
            repetitions: 0,
            queue_state: 'new',
          });
          await expect(flashcards.getManageFlashcardRow(card.uuid)).toBeVisible();
        }
        const baseline = await readApi<FlashcardAnalyticsSummary>(
          `/api/v1/flashcards/analytics/summary?deck_id=${deck.id}`
        );
        expect(baseline.reviewed_today).toBe(0);
        expect(baseline.decks).toEqual([
          expect.objectContaining({ deck_id: deck.id, total: 5, new: 5 }),
        ]);
        evidence.analyticsBefore = baseline;
      });

      await test.step('Review each distinct card Easy against its authoritative interval preview', async () => {
        await flashcards.switchToTab('study');
        await flashcards.selectReviewDeckByName(deckName);
        await flashcards.setReviewMode('due');
        const reviewedIds = new Set<string>();
        for (let index = 0; index < 5; index += 1) {
          const next = await readApi<FlashcardNextReviewResponse>(
            `/api/v1/flashcards/review/next?deck_id=${deck.id}`
          );
          expect(next.card).toBeTruthy();
          const card = next.card!;
          expect(savedCards.map((item) => item.uuid)).toContain(card.uuid);
          expect(reviewedIds.has(card.uuid)).toBe(false);
          await expect(flashcards.reviewActiveCard).toContainText(card.front);
          const sourceLink = flashcards.reviewActiveCard.getByRole('link', {
            name: `Note #${noteId}`,
            exact: true,
          });
          await expect(sourceLink).toHaveAttribute(
            'href',
            `/notes?source_ref_id=${encodeURIComponent(noteId)}`
          );
          if (index === 0) {
            await sourceLink.click();
            await expect(page).toHaveURL(
              new RegExp(`/notes\\?source_ref_id=${encodeURIComponent(noteId)}$`)
            );
            await notes.assertPageReady();
            await expect(notes.titleInput).toHaveValue(noteTitle);
            await notes.ensureMarkdownMode();
            await expect(notes.contentTextarea).toHaveValue(BIOLOGY_NOTE_CONTENT);
            await page.goBack();
            await flashcards.assertPageReady();
            await flashcards.switchToTab('study');
            await flashcards.selectReviewDeckByName(deckName);
            await expect(flashcards.reviewActiveCard).toContainText(card.front);
          }
          await flashcards.reviewShowAnswerButton.click();
          await expect(flashcards.reviewActiveCard).toContainText(card.back);
          expect(card.next_intervals?.easy).toMatch(/^\d+ days?$/);
          const previewDays = Number(card.next_intervals!.easy.match(/^\d+/)![0]);
          await expect(flashcards.reviewRateEasyButton).toContainText(card.next_intervals!.easy);
          const [{ request, response }] = await Promise.all([
            expectApiCall(page, { method: 'POST', url: /\/api\/v1\/flashcards\/review$/ }, 30_000),
            flashcards.reviewRateEasyButton.click(),
          ]);
          expect(request.postDataJSON()).toMatchObject({
            card_uuid: card.uuid,
            rating: 5,
            review_context: { deck_id: deck.id, review_mode: 'due' },
          });
          expect(response.ok()).toBe(true);
          const reviewed = (await response.json()) as FlashcardReviewResponse;
          expect(reviewed).toMatchObject({
            uuid: card.uuid,
            repetitions: 1,
            lapses: 0,
            queue_state: 'review',
            interval_days: previewDays,
            scheduler_type: deck.scheduler_type,
          });
          expect(reviewed.review_session_id).toBeGreaterThan(0);
          // Use server review time, not browser wall time. New-card Easy has no
          // interval fuzz; allow 2 seconds for serialization/time granularity.
          const intervalMs = Date.parse(reviewed.due_at!) - Date.parse(reviewed.last_reviewed_at!);
          expect(Math.abs(intervalMs - previewDays * 86_400_000)).toBeLessThanOrEqual(2_000);
          reviewedIds.add(card.uuid);
          reviewedCards.push(reviewed);
          if (index < 4) {
            await expect(flashcards.reviewActiveCard).not.toContainText(card.front);
            await expect(flashcards.reviewShowAnswerButton).toBeVisible();
          }
        }
        expect(reviewedIds.size).toBe(5);
        expect(new Set(reviewedCards.map((card) => card.review_session_id)).size).toBe(1);
        await expect(flashcards.reviewCompletionState).toContainText(
          /5 cards reviewed this session/i
        );
        // Wait for automatic completion to persist before reload can abort it.
        await expect
          .poll(async () => {
            const sessions = await readApi<FlashcardReviewSessionSummary[]>(
              `/api/v1/flashcards/review-sessions?deck_id=${deck.id}`
            );
            return sessions.find((session) => session.id === reviewedCards[0].review_session_id)
              ?.status;
          })
          .toBe('completed');
        evidence.reviewedCards = reviewedCards;
      });

      await test.step('Reload and verify saved schedules, completed session and exact analytics', async () => {
        await page.reload();
        await flashcards.assertPageReady();
        await flashcards.switchToTab('study');
        await flashcards.selectReviewDeckByName(deckName);
        await expect(flashcards.reviewCompletionState).toBeVisible();
        const stored = await readApi<FlashcardListResponse>(
          `/api/v1/flashcards?deck_id=${deck.id}&due_status=all&limit=100`
        );
        expect(stored.total).toBe(5);
        expect(stored.items).toHaveLength(5);
        for (const reviewed of reviewedCards) {
          expect(stored.items.find((card) => card.uuid === reviewed.uuid)).toMatchObject({
            uuid: reviewed.uuid,
            repetitions: 1,
            lapses: 0,
            queue_state: 'review',
            due_at: reviewed.due_at,
            last_reviewed_at: reviewed.last_reviewed_at,
            interval_days: reviewed.interval_days,
            source_ref_type: 'note',
            source_ref_id: noteId,
          });
        }
        expect(
          (
            await readApi<FlashcardNextReviewResponse>(
              `/api/v1/flashcards/review/next?deck_id=${deck.id}`
            )
          ).card
        ).toBeNull();
        const sessions = await readApi<FlashcardReviewSessionSummary[]>(
          `/api/v1/flashcards/review-sessions?deck_id=${deck.id}`
        );
        expect(sessions).toEqual([
          expect.objectContaining({
            id: reviewedCards[0].review_session_id,
            deck_id: deck.id,
            cards_reviewed: 5,
            status: 'completed',
          }),
        ]);
        const analytics = await readApi<FlashcardAnalyticsSummary>(
          `/api/v1/flashcards/analytics/summary?deck_id=${deck.id}`
        );
        expect(analytics).toMatchObject({
          reviewed_today: 5,
          retention_rate_today: 100,
          lapse_rate_today: 0,
        });
        expect(analytics.decks).toEqual([
          expect.objectContaining({ deck_id: deck.id, total: 5, new: 0, learning: 0, due: 0 }),
        ]);
        await expect(flashcards.reviewAnalyticsSummary).toContainText(/Reviewed today\s*5/);
        await expect(flashcards.reviewAnalyticsSummary).toContainText(/Retention rate\s*100\.0%/);
        await expect(flashcards.reviewAnalyticsSummary).toContainText(/Lapse rate\s*0\.0%/);
        evidence.sessions = sessions;
        evidence.analyticsAfter = analytics;
        expect(
          writes.filter((response) => new URL(response.url()).pathname.endsWith('/generate'))
        ).toHaveLength(1);
        expect(
          writes.filter((response) => new URL(response.url()).pathname.endsWith('/review'))
        ).toHaveLength(5);
        // Reopen the saved source URL already asserted on all five review cards.
        await page.goto(`/notes?source_ref_id=${encodeURIComponent(noteId)}`);
        await notes.assertPageReady();
        await expect(notes.titleInput).toHaveValue(noteTitle);
        await notes.ensureMarkdownMode();
        await expect(notes.contentTextarea).toHaveValue(BIOLOGY_NOTE_CONTENT);
      });
    } finally {
      page.off('response', captureWrite);
      evidence.writes = await Promise.all(
        writes.map(async (response) => ({
          path: new URL(response.url()).pathname,
          status: response.status(),
          request: response.request().postDataJSON(),
          response: await response.json().catch(() => null),
        }))
      );
      await testInfo.attach('notes-study-lineage.json', {
        body: JSON.stringify(evidence, null, 2),
        contentType: 'application/json',
      });
    }
  });
});
