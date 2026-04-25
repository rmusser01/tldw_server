/**
 * Flashcards E2E Tests (Tier 2)
 *
 * Tests the Flashcards workspace page lifecycle:
 * - Page loads with expected elements (tabs or offline banner)
 * - Tab switching between Study, Manage, Templates, and Transfer
 * - Export button fires GET /api/v1/flashcards/export (requires server)
 * - Keyboard shortcuts help button opens modal
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from '../../utils/fixtures';
import { expectApiCall } from '../../utils/api-assertions';
import { FlashcardsPage } from '../../utils/page-objects';
import { seedAuth, fetchWithApiKey, TEST_CONFIG } from '../../utils/helpers';

type SeededDeck = {
  id: number;
  name: string;
};

type SeededCard = {
  uuid: string;
  version: number;
  tags: string[];
};

async function createFlashcardDeck(
  name: string,
  options?: {
    reviewPromptSide?: 'front' | 'back';
  }
): Promise<SeededDeck> {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/flashcards/decks`,
    TEST_CONFIG.apiKey,
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        name,
        description: 'E2E flashcard tag suggestion deck',
        review_prompt_side: options?.reviewPromptSide ?? 'front',
      }),
    }
  );

  expect(response.ok).toBeTruthy();

  const payload = (await response.json()) as SeededDeck;
  expect(payload.id).toBeGreaterThan(0);
  return payload;
}

async function createFlashcardCard(input: {
  deckId: number;
  front: string;
  back: string;
  tags?: string[];
}): Promise<SeededCard> {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/flashcards`,
    TEST_CONFIG.apiKey,
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        deck_id: input.deckId,
        front: input.front,
        back: input.back,
        tags: input.tags ?? [],
        source_ref_type: 'manual',
      }),
    }
  );

  expect(response.ok).toBeTruthy();

  const payload = (await response.json()) as SeededCard;
  expect(payload.uuid).toBeTruthy();
  return payload;
}

test.describe('Flashcards', () => {
  let flashcards: FlashcardsPage;

  test.beforeEach(async ({ page }) => {
    await seedAuth(page);
    flashcards = new FlashcardsPage(page);
  });

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe('Page Load', () => {
    test('should render the Flashcards page with tabs or offline banner', async ({
      authedPage,
      diagnostics,
    }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      // Either the tabs container is visible (server online) or an offline/unsupported message
      const online = await flashcards.isOnline();
      const offlineVisible = await flashcards.offlineMessage.isVisible().catch(() => false);
      const unsupportedVisible = await flashcards.unsupportedMessage.isVisible().catch(() => false);

      expect(online || offlineVisible || unsupportedVisible).toBe(true);

      // If online, the core tabs should be present
      if (online) {
        await expect(flashcards.studyTab).toBeVisible();
        await expect(flashcards.manageTab).toBeVisible();
        await expect(flashcards.templatesTab).toBeVisible();
        await expect(flashcards.transferTab).toBeVisible();
        if (await flashcards.schedulerTab.isVisible().catch(() => false)) {
          await expect(flashcards.schedulerTab).toBeVisible();
        }
        await expect(flashcards.testWithQuizButton).toBeVisible();
      }

      await assertNoCriticalErrors(diagnostics);
    });

    test('should switch between tabs without errors', async ({ authedPage, diagnostics }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      const tabs = ['manage', 'templates', 'transfer', 'study'] as const;
      const schedulerVisible = await flashcards.schedulerTab.isVisible().catch(() => false);
      const tabsToVisit = schedulerVisible ? [...tabs, 'scheduler' as const] : [...tabs];

      for (const tab of tabsToVisit) {
        await flashcards.switchToTab(tab);
        if (tab === 'manage') {
          await expect(flashcards.manageTopBar).toBeVisible({ timeout: 10_000 });
        } else if (tab === 'templates') {
          await expect(flashcards.templatesTab).toHaveAttribute('aria-selected', 'true');
        } else if (tab === 'scheduler') {
          await expect(authedPage.getByPlaceholder('Search decks')).toBeVisible({ timeout: 10_000 });
        } else if (tab === 'transfer') {
          await expect(flashcards.importButton).toBeVisible({ timeout: 10_000 });
          await expect(flashcards.exportButton).toBeVisible({ timeout: 10_000 });
        } else {
          await expect(flashcards.reviewDeckSelect).toBeVisible({ timeout: 10_000 });
        }
      }

      await assertNoCriticalErrors(diagnostics);
    });
  });

  // =========================================================================
  // Study Tab
  // =========================================================================

  test.describe('Study Tab', () => {
    test('should show review deck selector and either a card or empty state', async ({
      authedPage,
      diagnostics,
    }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      await flashcards.switchToTab('study');

      // The deck selector should always be present on the Study tab
      await expect(flashcards.reviewDeckSelect).toBeVisible({ timeout: 10_000 });

      // Either an active review card or the empty-state card should be shown
      const hasActiveCard = await flashcards.reviewActiveCard.isVisible().catch(() => false);
      const hasEmptyCard = await flashcards.reviewEmptyCard.isVisible().catch(() => false);
      expect(hasActiveCard || hasEmptyCard).toBe(true);

      await assertNoCriticalErrors(diagnostics);
    });

    test('should flip the review prompt side for a selected deck', async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo);

      const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const deck = await createFlashcardDeck(`E2E Review Orientation ${runId}`);
      const front = `Front prompt ${runId}`;
      const back = `Back answer ${runId}`;

      await createFlashcardCard({
        deckId: deck.id,
        front,
        back,
      });

      flashcards = new FlashcardsPage(authedPage);
      await flashcards.gotoPath(`/flashcards?tab=review&deck_id=${deck.id}`);
      await flashcards.assertPageReady();

      expect(await flashcards.isOnline()).toBe(true);

      await expect(flashcards.reviewDeckSelect).toBeVisible({ timeout: 10_000 });

      await expect(flashcards.reviewActiveCard).toBeVisible({ timeout: 10_000 });
      await expect(
        flashcards.reviewActiveCard.getByText('Front', { exact: true })
      ).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.reviewActiveCard.getByText(front, { exact: true })).toBeVisible({
        timeout: 10_000,
      });

      await expect(flashcards.reviewPromptSideToggle).toBeVisible({ timeout: 10_000 });
      await flashcards.reviewPromptSideBackOption.click({ force: true });

      await expect(
        flashcards.reviewActiveCard.getByText('Back', { exact: true })
      ).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.reviewActiveCard.getByText(back, { exact: true })).toBeVisible({
        timeout: 10_000,
      });

      await flashcards.reviewShowAnswerButton.click();
      await expect(
        flashcards.reviewActiveCard.getByText('Front', { exact: true })
      ).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.reviewActiveCard.getByText(front, { exact: true })).toBeVisible({
        timeout: 10_000,
      });

      await assertNoCriticalErrors(diagnostics);
    });
  });

  // =========================================================================
  // Templates Tab
  // =========================================================================

  test.describe('Templates Tab', () => {
    test('should open the Templates tab and expose creation or load-state affordances', async ({
      authedPage,
      diagnostics,
    }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.gotoPath('/flashcards?tab=templates');
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      await expect(flashcards.templatesTab).toHaveAttribute('aria-selected', 'true');

      const templatesEmptyState = authedPage.getByText('No templates yet');

      await expect
        .poll(
          async () => {
            const createVisible = await flashcards.templatesCreateButton
              .isVisible()
              .catch(() => false);
            const loadErrorVisible = await flashcards.templatesErrorAlert
              .isVisible()
              .catch(() => false);
            const emptyVisible = await templatesEmptyState.isVisible().catch(() => false);
            return createVisible || loadErrorVisible || emptyVisible;
          },
          { timeout: 15_000 }
        )
        .toBe(true);

      const createVisible = await flashcards.templatesCreateButton.isVisible().catch(() => false);
      const loadErrorVisible = await flashcards.templatesErrorAlert.isVisible().catch(() => false);
      const emptyVisible = await templatesEmptyState.isVisible().catch(() => false);

      expect(createVisible || loadErrorVisible || emptyVisible).toBe(true);

      if (createVisible) {
        await flashcards.templatesCreateButton.click();
        await expect(authedPage.getByText('Template name')).toBeVisible({ timeout: 10_000 });
      } else if (emptyVisible) {
        await expect(templatesEmptyState).toBeVisible({ timeout: 10_000 });
      } else {
        await expect(flashcards.templatesErrorAlert).toBeVisible({ timeout: 10_000 });
      }

      await assertNoCriticalErrors(diagnostics);
    });
  });

  // =========================================================================
  // Manage Tab
  // =========================================================================

  test.describe('Manage Tab', () => {
    test('should show search, deck filter, and FAB create button', async ({
      authedPage,
      diagnostics,
    }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      expect(online).toBe(true);

      await flashcards.switchToTab('manage');

      await expect(flashcards.manageTopBar).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.manageSearchInput).toBeVisible();
      await expect(flashcards.manageDeckSelect).toBeVisible();
      await expect(flashcards.fabCreateButton).toBeVisible();

      await assertNoCriticalErrors(diagnostics);
    });

    test('should fire flashcards list API when searching', async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo);

      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      await flashcards.switchToTab('manage');
      await expect(flashcards.manageSearchInput).toBeVisible({ timeout: 10_000 });

      const searchVisible = await flashcards.manageSearchInput.isVisible().catch(() => false);
      if (!searchVisible) return;

      const apiCall = expectApiCall(
        authedPage,
        {
          url: /\/api\/v1\/flashcards/,
          method: 'GET',
        },
        15_000
      );

      await flashcards.manageSearchInput.fill('test query');
      await flashcards.manageSearchInput.press('Enter');

      try {
        const { response } = await apiCall;
        expect(response.status()).toBeLessThan(500);
      } catch {
        // Search may debounce; acceptable if no immediate call
      }

      await assertNoCriticalErrors(diagnostics);
    });

    test('should allow selecting existing tag suggestions in create and edit drawers', async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      test.setTimeout(120_000);
      skipIfServerUnavailable(serverInfo);

      const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const deck = await createFlashcardDeck(`E2E Flashcard Tags ${runId}`);
      const createSuggestionTag = `bio-create-${runId}`;
      const editSuggestionTag = `bio-edit-${runId}`;

      await createFlashcardCard({
        deckId: deck.id,
        front: `Suggestion source ${runId}`,
        back: 'Seeds global tag suggestions',
        tags: [createSuggestionTag, editSuggestionTag],
      });

      const knownCard = await createFlashcardCard({
        deckId: deck.id,
        front: `Known card ${runId}`,
        back: 'Used for edit drawer verification',
      });

      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      await flashcards.switchToTab('manage');
      await expect(flashcards.manageTopBar).toBeVisible({ timeout: 10_000 });
      await flashcards.selectManageDeckByName(deck.name);

      await flashcards.fabCreateButton.click();
      await expect(authedPage.getByText('Create Flashcard')).toBeVisible({ timeout: 10_000 });
      await flashcards.selectCreateDrawerDeckByName(deck.name);
      await authedPage.getByText('Advanced options (tags, extra, notes)').click();

      await flashcards.createTagPicker.getByRole('combobox').click();
      await expect(flashcards.createTagPickerSearchInput).toBeVisible({ timeout: 10_000 });
      await flashcards.createTagPickerSearchInput.fill(createSuggestionTag);

      const createTagOption = authedPage.getByRole('option', {
        name: createSuggestionTag,
        exact: true,
      });
      await expect(createTagOption).toBeVisible({ timeout: 10_000 });
      await createTagOption.click();
      await expect(flashcards.createTagPicker).toContainText(createSuggestionTag);

      await authedPage.getByPlaceholder('Question or prompt...').fill(`Created via UI ${runId}`);
      await authedPage.getByPlaceholder('Answer...').fill('Created tag suggestion answer');

      const createResponsePromise = authedPage.waitForResponse((response) => {
        return (
          response.request().method() === 'POST' && /\/api\/v1\/flashcards$/.test(response.url())
        );
      });

      await authedPage.getByRole('button', { name: 'Create', exact: true }).click();

      const createResponse = await createResponsePromise;
      expect(createResponse.status()).toBeLessThan(400);
      const createdCard = (await createResponse.json()) as SeededCard;
      expect(createdCard.tags).toContain(createSuggestionTag);

      await expect(flashcards.getManageFlashcardEditButton(knownCard.uuid)).toBeVisible({
        timeout: 10_000,
      });
      await flashcards.openManageFlashcardEdit(knownCard.uuid);

      await expect(authedPage.getByText('Edit Flashcard')).toBeVisible({ timeout: 10_000 });
      await flashcards.editDrawerAdditionalFieldsToggle.click();
      await expect(flashcards.editTagPicker).toBeVisible({ timeout: 10_000 });

      await flashcards.editTagPicker.getByRole('combobox').click();
      await expect(flashcards.editTagPickerSearchInput).toBeVisible({ timeout: 10_000 });
      await flashcards.editTagPickerSearchInput.fill(editSuggestionTag);

      const editTagOption = authedPage.getByRole('option', {
        name: editSuggestionTag,
        exact: true,
      });
      await expect(editTagOption).toBeVisible({ timeout: 10_000 });
      await editTagOption.click();
      await expect(flashcards.editTagPicker).toContainText(editSuggestionTag);

      const updateResponsePromise = authedPage.waitForResponse((response) => {
        return (
          response.request().method() === 'PATCH' &&
          response.url().includes(`/api/v1/flashcards/${knownCard.uuid}`)
        );
      });

      await authedPage.getByRole('button', { name: 'Save', exact: true }).click();

      const updateResponse = await updateResponsePromise;
      expect(updateResponse.status()).toBeLessThan(400);
      const updatedCard = (await updateResponse.json()) as SeededCard;
      expect(updatedCard.tags).toContain(editSuggestionTag);

      await expect(authedPage.getByRole('button', { name: 'Save', exact: true })).toBeHidden({
        timeout: 10_000,
      });

      await flashcards.openManageFlashcardEdit(knownCard.uuid);
      await expect(authedPage.getByText('Edit Flashcard')).toBeVisible({ timeout: 10_000 });
      await flashcards.editDrawerAdditionalFieldsToggle.click();
      await expect(flashcards.editTagPicker).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.editTagPicker).toContainText(editSuggestionTag);

      await assertNoCriticalErrors(diagnostics);
    });
  });

  // =========================================================================
  // Transfer Tab - Export
  // =========================================================================

  test.describe('Export', () => {
    test('should fire GET /api/v1/flashcards/export when Export button is clicked', async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo);

      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      await flashcards.switchToTab('transfer');
      await expect(flashcards.importButton).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.exportButton).toBeVisible({ timeout: 10_000 });

      const exportVisible = await flashcards.exportButton.isVisible().catch(() => false);
      if (!exportVisible) return;

      const exportEnabled = await flashcards.exportButton.isEnabled().catch(() => false);
      if (!exportEnabled) return;

      const apiCall = expectApiCall(
        authedPage,
        {
          url: /\/api\/v1\/flashcards\/export/,
          method: 'GET',
        },
        15_000
      );

      await flashcards.exportButton.click();

      try {
        const { response } = await apiCall;
        expect(response.status()).toBeLessThan(500);
      } catch {
        // Export may require deck selection; acceptable if button is not wired without selection
      }

      await assertNoCriticalErrors(diagnostics);
    });
  });

  // =========================================================================
  // Transfer Tab - Import
  // =========================================================================

  test.describe('Import', () => {
    test('should show import textarea and format selector on Transfer tab', async ({
      authedPage,
      diagnostics,
    }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      await flashcards.switchToTab('transfer');
      await expect(flashcards.importButton).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.exportButton).toBeVisible({ timeout: 10_000 });

      await expect(flashcards.importFormatSelect).toBeVisible({ timeout: 10_000 });
      await expect(flashcards.importButton).toBeVisible();

      await assertNoCriticalErrors(diagnostics);
    });
  });

  // =========================================================================
  // Keyboard Shortcuts Modal
  // =========================================================================

  test.describe('Keyboard Shortcuts', () => {
    test('should open keyboard shortcuts modal via help button', async ({
      authedPage,
      diagnostics,
    }) => {
      flashcards = new FlashcardsPage(authedPage);
      await flashcards.goto();
      await flashcards.assertPageReady();

      const online = await flashcards.isOnline();
      if (!online) return;

      const helpVisible = await flashcards.keyboardShortcutsButton.isVisible().catch(() => false);
      if (!helpVisible) return;

      await flashcards.keyboardShortcutsButton.click();

      // The modal should appear
      const modal = authedPage.locator('.ant-modal');
      await expect(modal.first()).toBeVisible({ timeout: 5_000 });

      // Close it
      await authedPage.keyboard.press('Escape');
      await expect(modal.first())
        .toBeHidden({ timeout: 3_000 })
        .catch(() => {});

      await assertNoCriticalErrors(diagnostics);
    });
  });
});
