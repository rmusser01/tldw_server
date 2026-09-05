/**
 * Journey: Notes -> Flashcards
 *
 * End-to-end workflow that creates a note, then generates flashcards
 * from the note content, and verifies flashcards were created.
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels } from '../../utils/fixtures';
import { NotesPage, FlashcardsPage } from '../../utils/page-objects';
import { expectApiCall } from '../../utils/api-assertions';
import { createNote } from '../../utils/journey-helpers';
import { fetchWithApiKey, generateTestId, TEST_CONFIG } from '../../utils/helpers';

type FlashcardDeck = {
  id: number;
  name: string;
  version: number;
};

async function createJourneyDeck(name: string): Promise<FlashcardDeck> {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/flashcards/decks`,
    TEST_CONFIG.apiKey,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        name,
        description: 'Task2C isolated flashcard journey deck',
      }),
    }
  );
  expect(response.ok).toBeTruthy();
  return (await response.json()) as FlashcardDeck;
}

async function cleanupJourneyDeck(deck: FlashcardDeck): Promise<void> {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/flashcards/decks/${deck.id}?expected_version=${deck.version}`,
    TEST_CONFIG.apiKey,
    { method: 'DELETE' }
  );
  if (!response.ok && response.status !== 404 && response.status !== 409) {
    throw new Error(
      `Failed to cleanup Task2C flashcard deck ${deck.id}: ${response.status} ${response.statusText}`
    );
  }
}

test.describe('Notes -> Flashcards journey', () => {
  const noteTitle = `E2E-Study-Note-${Date.now()}`;
  const noteContent = [
    'The mitochondria is the powerhouse of the cell.',
    'DNA stands for deoxyribonucleic acid.',
    'Photosynthesis converts light energy into chemical energy.',
    'The human body has 206 bones.',
    'Water boils at 100 degrees Celsius at sea level.',
  ].join('\n\n');

  test('create note and generate flashcards from it', async ({
    authedPage: page,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo);
    skipIfNoModels(serverInfo);

    const deck = await createJourneyDeck(generateTestId('task2c-flashcards'));

    try {
      await test.step('Create a note with study content', async () => {
        // Use the journey helper to create the note
        const title = await createNote(page, {
          title: noteTitle,
          content: noteContent,
        });
        expect(title).toBe(noteTitle);
      });

      await test.step('Verify note was saved', async () => {
        const notesPage = new NotesPage(page);
        await notesPage.goto();
        await notesPage.assertPageReady();
        await notesPage.assertNoteVisible(noteTitle);
      });

      await test.step('Navigate to flashcards and generate from content', async () => {
        const flashcardsPage = new FlashcardsPage(page);
        await flashcardsPage.goto();
        await flashcardsPage.assertPageReady();

        // Check if flashcards feature is available
        const isOnline = await flashcardsPage.isOnline();
        if (!isOnline) {
          test.skip(true, 'Flashcards feature not available');
          return;
        }

        // Switch to the transfer tab to access the generate feature
        await flashcardsPage.switchToTab('transfer');

        await page.getByTestId('flashcards-generate-deck').click();
        await page.getByText(deck.name, { exact: true }).last().click();
        await expect(page.getByTestId('flashcards-generate-deck')).toContainText(deck.name);

        const exportPreview = page.getByText(/\d+\s+cards from All decks/i).first();
        const initialExportPreview = await exportPreview.textContent();
        const initialCardCount = Number(initialExportPreview?.match(/(\d+)\s+cards/i)?.[1] ?? '0');

        // Check if the generate textarea is available
        const generateVisible = await flashcardsPage.generateTextarea
          .isVisible()
          .catch(() => false);

        if (!generateVisible) {
          test.skip(true, 'Flashcard generation feature not available in UI');
          return;
        }

        // Paste the note content into the generate textarea
        await flashcardsPage.generateTextarea.fill(noteContent);

        // Click generate button
        const generateBtnVisible = await flashcardsPage.generateButton
          .isVisible()
          .catch(() => false);

        if (generateBtnVisible) {
          const generateApiCall = expectApiCall(
            page,
            {
              method: 'POST',
              url: '/api/v1/flashcards/generate',
            },
            60_000
          );

          await flashcardsPage.generateButton.click();

          const { response } = await generateApiCall;
          expect(response.status()).toBeLessThan(400);
          const responseBody = await response.json().catch(() => ({}));
          expect(Number(responseBody?.count ?? 0)).toBeGreaterThan(0);

          const saveGeneratedButton = page.getByTestId('flashcards-generate-save-button');
          await expect(saveGeneratedButton).toBeVisible({ timeout: 15_000 });

          const saveApiCall = expectApiCall(
            page,
            {
              method: 'POST',
              url: /\/api\/v1\/flashcards(?:\?|$)/,
            },
            60_000
          );

          await saveGeneratedButton.click();

          const { request: saveRequest, response: saveResponse } = await saveApiCall;
          expect(saveResponse.status()).toBeLessThan(400);
          expect((saveRequest.postDataJSON() as { deck_id?: unknown }).deck_id).toBe(deck.id);
          await expect(
            page.getByText(/Saved \d+ (?:generated )?cards(?:; \d+ failed\.)?/i).first()
          ).toBeVisible({ timeout: 15_000 });
          await expect
            .poll(
              async () => {
                const updatedPreview = await exportPreview.textContent();
                return Number(updatedPreview?.match(/(\d+)\s+cards/i)?.[1] ?? '0');
              },
              { timeout: 15_000 }
            )
            .toBeGreaterThan(initialCardCount);
        }
      });
    } finally {
      await cleanupJourneyDeck(deck);
    }
  });
});
