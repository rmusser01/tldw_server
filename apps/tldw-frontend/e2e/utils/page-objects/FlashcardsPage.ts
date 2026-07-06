/**
 * Page Object for the Flashcards workspace
 *
 * The route renders FlashcardsWorkspace which shows either:
 * - A connection/offline banner when the server is unreachable
 * - FlashcardsManager with tabs for Study, Manage, Transfer, Templates, and Scheduler
 *
 * API base paths:
 *   /api/v1/flashcards        (cards CRUD, review, generate, import, export)
 *   /api/v1/flashcards/decks  (deck CRUD)
 */
import { type APIRequestContext, type Page, type Locator, expect } from '@playwright/test';
import { BasePage, type InteractiveElement } from './BasePage';
import {
  waitForAppShell,
  waitForConnection,
  dismissConnectionModals,
  TEST_CONFIG,
} from '../helpers';

export const FLASHCARDS_E2E_PREFIX = 'codex-flashcards-ux';

function hashFlashcardsRunSeed(seed: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

export function makeFlashcardsRunId(seed: string): string {
  const normalizedSeed = seed
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
  return `${FLASHCARDS_E2E_PREFIX}-${normalizedSeed || 'run'}-${hashFlashcardsRunSeed(seed)}`;
}

export type FlashcardsSeedRecord = {
  runId: string;
  deckName: string;
  cardFront: string;
  cardBack: string;
};

export function buildFlashcardsSeedRecord(
  runId: string
): FlashcardsSeedRecord {
  return {
    runId,
    deckName: `${runId}-deck`,
    cardFront: `${runId} front`,
    cardBack: `${runId} back`,
  };
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

type FlashcardsCleanupDeck = {
  id?: number;
  name?: string;
  version?: number;
};

type FlashcardsCleanupCard = {
  uuid?: string;
  front?: string | null;
  back?: string | null;
  tags?: string[] | null;
  version?: number;
};

function flashcardsApiUrl(path: string): string {
  return `${TEST_CONFIG.serverUrl.replace(/\/$/, '')}${path}`;
}

function flashcardsApiHeaders(): Record<string, string> {
  return {
    'X-API-Key': TEST_CONFIG.apiKey,
  };
}

export async function assertFlashcardsBackendReady(
  request: APIRequestContext
): Promise<void> {
  const response = await request.get(flashcardsApiUrl('/api/v1/health'), {
    headers: flashcardsApiHeaders(),
  });

  if (!response.ok()) {
    throw new Error(
      `Flashcards backend preflight failed: ${response.status()} ${await response.text()}`
    );
  }

  const flashcardsResponse = await request.get(
    flashcardsApiUrl('/api/v1/flashcards/decks?limit=1&include_deleted=false'),
    { headers: flashcardsApiHeaders() }
  );

  if (!flashcardsResponse.ok()) {
    throw new Error(
      `Flashcards API preflight failed: ${flashcardsResponse.status()} ${await flashcardsResponse.text()}`
    );
  }
}

const startsWithFlashcardsRunId = (value: unknown, runId: string): boolean =>
  typeof value === 'string' && value.startsWith(runId);

function warnFlashcardsCleanup(message: string, error?: unknown): void {
  if (error === undefined) {
    console.warn(`[flashcards:e2e:cleanup] ${message}`);
    return;
  }
  console.warn(`[flashcards:e2e:cleanup] ${message}`, error);
}

async function cleanupFlashcardCard(
  request: APIRequestContext,
  card: FlashcardsCleanupCard
): Promise<void> {
  if (!card.uuid || typeof card.version !== 'number') return;

  try {
    const response = await request.delete(
      flashcardsApiUrl(
        `/api/v1/flashcards/${encodeURIComponent(card.uuid)}?expected_version=${card.version}`
      ),
      { headers: flashcardsApiHeaders() }
    );

    if (!response.ok() && response.status() !== 404 && response.status() !== 409) {
      warnFlashcardsCleanup(
        `Failed to cleanup flashcard card ${card.uuid}: ${response.status()} ${await response.text()}`
      );
    }
  } catch (error) {
    warnFlashcardsCleanup(`Failed to cleanup flashcard card ${card.uuid}`, error);
  }
}

async function cleanupFlashcardDeck(
  request: APIRequestContext,
  deck: FlashcardsCleanupDeck
): Promise<void> {
  if (typeof deck.id !== 'number' || typeof deck.version !== 'number') return;

  try {
    const response = await request.delete(
      flashcardsApiUrl(
        `/api/v1/flashcards/decks/${deck.id}?expected_version=${deck.version}`
      ),
      { headers: flashcardsApiHeaders() }
    );

    if (!response.ok() && response.status() !== 404 && response.status() !== 409) {
      warnFlashcardsCleanup(
        `Failed to cleanup flashcard deck ${deck.id}: ${response.status()} ${await response.text()}`
      );
    }
  } catch (error) {
    warnFlashcardsCleanup(`Failed to cleanup flashcard deck ${deck.id}`, error);
  }
}

export async function cleanupFlashcardsRunRecords(
  request: APIRequestContext,
  runId: string
): Promise<void> {
  try {
    await assertFlashcardsBackendReady(request);
  } catch (error) {
    warnFlashcardsCleanup(`Skipping cleanup for ${runId}; backend preflight failed`, error);
    return;
  }

  let cardResponse;
  try {
    cardResponse = await request.get(
      flashcardsApiUrl(
        `/api/v1/flashcards?q=${encodeURIComponent(runId)}&limit=1000&due_status=all`
      ),
      { headers: flashcardsApiHeaders() }
    );
  } catch (error) {
    warnFlashcardsCleanup(`Failed to list flashcards for cleanup run ${runId}`, error);
    return;
  }

  if (cardResponse.ok()) {
    const payload = (await cardResponse.json().catch(() => ({}))) as {
      items?: FlashcardsCleanupCard[];
    };
    const cards = Array.isArray(payload.items) ? payload.items : [];
    await Promise.allSettled(
      cards
        .filter(
          card =>
            startsWithFlashcardsRunId(card.front, runId) ||
            startsWithFlashcardsRunId(card.back, runId) ||
            card.tags?.some(tag => startsWithFlashcardsRunId(tag, runId))
        )
        .map(card => cleanupFlashcardCard(request, card))
    );
  } else {
    warnFlashcardsCleanup(
      `Failed to list flashcards for cleanup run ${runId}: ${cardResponse.status()} ${await cardResponse.text()}`
    );
  }

  let deckResponse;
  try {
    deckResponse = await request.get(
      flashcardsApiUrl('/api/v1/flashcards/decks?limit=1000&include_deleted=false'),
      { headers: flashcardsApiHeaders() }
    );
  } catch (error) {
    warnFlashcardsCleanup(`Failed to list decks for cleanup run ${runId}`, error);
    return;
  }

  if (!deckResponse.ok()) {
    warnFlashcardsCleanup(
      `Failed to list decks for cleanup run ${runId}: ${deckResponse.status()} ${await deckResponse.text()}`
    );
    return;
  }

  const decks = (await deckResponse.json().catch(() => [])) as FlashcardsCleanupDeck[];
  if (!Array.isArray(decks)) return;

  await Promise.allSettled(
    decks
      .filter(deck => startsWithFlashcardsRunId(deck.name, runId))
      .map(deck => cleanupFlashcardDeck(request, deck))
  );
}

export class FlashcardsPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto('/flashcards', { waitUntil: 'domcontentloaded' });
    await waitForConnection(this.page);
  }

  async gotoPath(path: string): Promise<void> {
    await this.page.goto(path, { waitUntil: 'domcontentloaded' });
    await waitForConnection(this.page);
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000);
    // Either the tabs container is visible (online) or a connection banner
    const tabs = this.page.locator('[data-testid="flashcards-tabs"]');
    const offline = this.page.getByText('Connect to use Flashcards');
    const unsupported = this.page.getByText('Flashcards API not available');
    await Promise.race([
      tabs.waitFor({ state: 'visible', timeout: 20_000 }),
      offline.first().waitFor({ state: 'visible', timeout: 20_000 }),
      unsupported.first().waitFor({ state: 'visible', timeout: 20_000 }),
    ]).catch(() => {});
  }

  // -- Locators: Top-level ---------------------------------------------------

  /** The Ant Design Tabs container wrapping the flashcards workspace tabs */
  get tabsContainer(): Locator {
    return this.page.locator('[data-testid="flashcards-tabs"]');
  }

  /** Offline / not-connected banner */
  get offlineMessage(): Locator {
    return this.page.getByText('Connect to use Flashcards');
  }

  /** Feature-unavailable banner */
  get unsupportedMessage(): Locator {
    return this.page.getByText('Flashcards API not available');
  }

  // -- Locators: Tab buttons -------------------------------------------------

  get studyTab(): Locator {
    return this.page.getByRole('tab', { name: /study/i });
  }

  get manageTab(): Locator {
    return this.page.getByRole('tab', { name: /manage/i });
  }

  get templatesTab(): Locator {
    return this.page.getByRole('tab', { name: /templates/i });
  }

  get schedulerTab(): Locator {
    return this.page.getByRole('tab', { name: /scheduler/i });
  }

  get schedulerEmptyPreview(): Locator {
    return this.page.locator('[data-testid="flashcards-scheduler-empty-preview"]');
  }

  get templatesCreateButton(): Locator {
    return this.page.getByRole('button', { name: /create template/i });
  }

  get templatesErrorAlert(): Locator {
    return this.page.getByRole('alert').filter({ hasText: /could not load templates/i });
  }

  get transferTab(): Locator {
    return this.page.getByRole('tab', { name: /transfer|create\s*&\s*import|import\s*\/\s*export/i });
  }

  // -- Locators: Tab bar extra content ---------------------------------------

  /** "Test with Quiz" CTA button in the tab bar */
  get testWithQuizButton(): Locator {
    return this.page.locator('[data-testid="flashcards-to-quiz-cta"]');
  }

  /** Keyboard shortcuts help button (icon-only) */
  get keyboardShortcutsButton(): Locator {
    return this.page.getByRole('button', { name: /keyboard shortcuts/i });
  }

  // -- Locators: Study (Review) tab ------------------------------------------

  get reviewDeckSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-review-deck-select"]');
  }

  get reviewTopbar(): Locator {
    return this.page.locator('[data-testid="flashcards-review-topbar"]');
  }

  get reviewModeToggle(): Locator {
    return this.page.locator('[data-testid="flashcards-review-mode-toggle"]');
  }

  get reviewModeCramOption(): Locator {
    return this.reviewCramModeOption;
  }

  get reviewDueOnlyModeOption(): Locator {
    return this.reviewModeToggle.getByText('Due only', { exact: true });
  }

  get reviewCramModeOption(): Locator {
    return this.reviewModeToggle.getByText('Cram', { exact: true });
  }

  get reviewCramTagInput(): Locator {
    return this.page.locator('[data-testid="flashcards-review-cram-tag"]');
  }

  get reviewCramUpdateScheduleToggle(): Locator {
    return this.page.locator('[data-testid="flashcards-review-cram-update-schedule"]');
  }

  get reviewPromptSideToggle(): Locator {
    return this.page.locator('[data-testid="flashcards-review-prompt-side-toggle"]');
  }

  get reviewPromptSideFrontOption(): Locator {
    return this.reviewPromptSideToggle.getByText('Front first', { exact: true });
  }

  get reviewPromptSideBackOption(): Locator {
    return this.reviewPromptSideToggle.getByText('Back first', { exact: true });
  }

  get reviewActiveCard(): Locator {
    return this.page.locator('[data-testid="flashcards-review-active-card"]');
  }

  get reviewShowAnswerButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-show-answer"]');
  }

  get reviewGoodButton(): Locator {
    return this.reviewRateGoodButton;
  }

  get reviewEasyButton(): Locator {
    return this.reviewRateEasyButton;
  }

  get reviewRateAgainButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-rate-1"]');
  }

  get reviewRateHardButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-rate-2"]');
  }

  get reviewRateGoodButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-rate-3"]');
  }

  get reviewRateEasyButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-rate-4"]');
  }

  get reviewRetryAlert(): Locator {
    return this.page.locator('[data-testid="flashcards-review-retry-alert"]');
  }

  get reviewEndSessionButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-end-session"]');
  }

  get reviewEndSessionEmptyButton(): Locator {
    return this.page.locator('[data-testid="flashcards-review-end-session-empty"]');
  }

  get reviewShortcutQuestionChips(): Locator {
    return this.page.locator('[data-testid="flashcards-review-shortcut-chips-question"]');
  }

  get reviewShortcutAnswerChips(): Locator {
    return this.page.locator('[data-testid="flashcards-review-shortcut-chips-answer"]');
  }

  get reviewProgressStatus(): Locator {
    return this.page.getByRole('status').filter({ hasText: /cards remaining|reviewed|min left/i });
  }

  get reviewEmptyCard(): Locator {
    return this.page.locator('[data-testid="flashcards-review-empty-card"]');
  }

  get reviewCompletionState(): Locator {
    return this.reviewEmptyCard.filter({
      hasText: /cards reviewed this session|no cards are due|all caught up|cram complete/i,
    });
  }

  get reviewAnalyticsSummary(): Locator {
    return this.page.locator('[data-testid="flashcards-review-analytics-summary"]');
  }

  get reviewCreateCta(): Locator {
    return this.page.locator('[data-testid="flashcards-review-empty-create-cta"]');
  }

  get reviewImportCta(): Locator {
    return this.page.locator('[data-testid="flashcards-review-empty-import-cta"]');
  }

  get recentStudySessions(): Locator {
    return this.page.getByText('Recent study sessions', { exact: true });
  }

  // -- Locators: Manage tab --------------------------------------------------

  get manageTopBar(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-topbar"]');
  }

  get manageSearchInput(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-search"] input');
  }

  get manageDeckSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-deck-select"]');
  }

  get manageDueStatusFilter(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-due-status"]');
  }

  get manageSortSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-sort-select"]');
  }

  get manageShowWorkspaceDecksToggle(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-show-workspace-decks"]');
  }

  get manageWorkspaceFilter(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-workspace-filter"]');
  }

  get manageMoveScopeButton(): Locator {
    return this.page.locator('[data-testid="flashcards-manage-move-scope"]');
  }

  get fabCreateButton(): Locator {
    return this.page.locator('[data-testid="flashcards-fab-create"]');
  }

  get createDrawer(): Locator {
    return this.page.getByRole('dialog', { name: 'Create Flashcard' }).last();
  }

  get createDrawerDeckSelect(): Locator {
    return this.createDrawer.locator('.ant-select').first();
  }

  get createFrontTextarea(): Locator {
    return this.createDrawerFrontTextarea;
  }

  get createBackTextarea(): Locator {
    return this.createDrawerBackTextarea;
  }

  get createSubmitButton(): Locator {
    return this.createDrawerCreateButton;
  }

  get createAndAddAnotherButton(): Locator {
    return this.createDrawerCreateAndAddAnotherButton;
  }

  get createSuccessMessage(): Locator {
    return this.page.locator('.ant-message-notice-success').filter({ hasText: /created/i });
  }

  get createErrorMessage(): Locator {
    return this.page.locator('.ant-message-notice-error');
  }

  get createDrawerFrontTextarea(): Locator {
    return this.createDrawer.getByPlaceholder('Question or prompt...');
  }

  get createDrawerBackTextarea(): Locator {
    return this.createDrawer.getByPlaceholder('Answer...');
  }

  get createDrawerCreateButton(): Locator {
    return this.createDrawer.getByRole('button', { name: 'Create', exact: true });
  }

  get createDrawerCreateAndAddAnotherButton(): Locator {
    return this.createDrawer.getByRole('button', { name: 'Create & Add Another' });
  }

  get editDrawer(): Locator {
    return this.page.getByRole('dialog', { name: 'Edit Card' }).last();
  }

  get editDrawerAdditionalFieldsToggle(): Locator {
    return this.editDrawer.getByText('Additional fields', { exact: true });
  }

  get createTagPicker(): Locator {
    return this.createDrawer.locator('[data-testid="flashcards-create-tag-picker"]');
  }

  get createTagPickerSearchInput(): Locator {
    return this.createDrawer.locator('[data-testid="flashcards-create-tag-picker-search-input"]');
  }

  get editTagPicker(): Locator {
    return this.editDrawer.locator('[data-testid="flashcards-edit-tag-picker"]');
  }

  get editTagPickerSearchInput(): Locator {
    return this.editDrawer.locator('[data-testid="flashcards-edit-tag-picker-search-input"]');
  }

  // -- Locators: Transfer (Import/Export) tab --------------------------------

  get importFormatSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-import-format"]');
  }

  get importTaskPanel(): Locator {
    return this.page.locator('[data-testid="flashcards-import-task-panel"]');
  }

  get exportTaskPanel(): Locator {
    return this.page.locator('[data-testid="flashcards-export-task-panel"]');
  }

  get transferTaskSwitcher(): Locator {
    return this.page.locator('[data-testid="flashcards-transfer-task-switcher"]');
  }

  get transferImportTask(): Locator {
    return this.transferTaskSwitcher.getByText('Import file', { exact: true });
  }

  get transferExportTask(): Locator {
    return this.transferTaskSwitcher.getByText('Export backup', { exact: true });
  }

  get importTextarea(): Locator {
    return this.importTaskPanel.locator('[data-testid="flashcards-import-textarea"]');
  }

  get importDelimiterSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-import-delimiter"]');
  }

  get importPreflightWarning(): Locator {
    return this.page.locator('[data-testid="flashcards-import-preflight-warning"]');
  }

  get importButton(): Locator {
    return this.page.locator('[data-testid="flashcards-import-button"]');
  }

  get importResultAlert(): Locator {
    return this.page.locator('[data-testid="flashcards-import-last-result"]');
  }

  get structuredImportPreviewButton(): Locator {
    return this.page.locator('[data-testid="flashcards-structured-preview-button"]');
  }

  get structuredImportErrors(): Locator {
    return this.page.locator('[data-testid="flashcards-structured-preview-errors"]');
  }

  get structuredImportSaveButton(): Locator {
    return this.page.locator('[data-testid="flashcards-structured-save-button"]');
  }

  get exportDeckSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-export-deck"]');
  }

  get exportFormatSelect(): Locator {
    return this.page.locator('[data-testid="flashcards-export-format"]');
  }

  get exportButton(): Locator {
    return this.page.locator('[data-testid="flashcards-export-button"]');
  }

  get generateTextarea(): Locator {
    return this.page.locator('[data-testid="flashcards-generate-text"]');
  }

  get generateButton(): Locator {
    return this.page.locator('[data-testid="flashcards-generate-button"]');
  }

  getManageFlashcardRow(cardUuid: string): Locator {
    return this.page.locator(`[data-testid="flashcard-item-${cardUuid}"]`);
  }

  getManageFlashcardEditButton(cardUuid: string): Locator {
    return this.page.locator(`[data-testid="flashcard-edit-${cardUuid}"]`);
  }

  getActiveSelectOption(optionName: string, exact = false): Locator {
    const optionText = exact
      ? new RegExp(`^\\s*${escapeRegExp(optionName)}\\s*$`)
      : optionName;

    return this.page
      .locator('.ant-select-dropdown:not(.ant-select-dropdown-hidden):visible .ant-select-item-option-content')
      .filter({ hasText: optionText })
      .first();
  }

  async selectManageDeckByName(deckName: string): Promise<void> {
    const selectedDeck = this.manageDeckSelect.getByText(deckName, { exact: true });
    if (await selectedDeck.isVisible().catch(() => false)) return;
    await this.manageDeckSelect.click({ force: true });
    const deckOption = this.getActiveSelectOption(deckName);
    await expect(deckOption).toBeVisible({ timeout: 10_000 });
    await deckOption.click();
  }

  async selectReviewDeckByName(deckName: string): Promise<void> {
    await this.reviewDeckSelect.click({ force: true });
    const deckOption = this.getActiveSelectOption(deckName);
    await expect(deckOption).toBeVisible({ timeout: 10_000 });
    await deckOption.click();
  }

  async selectManageWorkspaceById(workspaceId: string): Promise<void> {
    await this.manageWorkspaceFilter.click({ force: true });
    const option = this.getActiveSelectOption(workspaceId, true);
    await expect(option).toBeVisible({ timeout: 10_000 });
    await option.click();
  }

  async selectFirstManageDeckOption(): Promise<void> {
    await this.manageDeckSelect.click({ force: true });
    await this.page.keyboard.press('ArrowDown');
    await this.page.keyboard.press('Enter');
  }

  async selectCreateDrawerDeckByName(deckName: string): Promise<void> {
    const selectedDeck = this.createDrawerDeckSelect.getByText(deckName, { exact: true });
    if (await selectedDeck.isVisible().catch(() => false)) return;
    await this.createDrawerDeckSelect.scrollIntoViewIfNeeded();
    await this.createDrawerDeckSelect.click({ force: true });
    const deckOption = this.getActiveSelectOption(deckName);
    await expect(deckOption).toBeVisible({ timeout: 10_000 });
    await deckOption.click();
  }

  async selectVisibleDropdownOption(optionName: string): Promise<void> {
    const option = this.getActiveSelectOption(optionName, true);
    await expect(option).toBeVisible({ timeout: 10_000 });
    await option.click();
  }

  async selectImportFormat(formatLabel: string): Promise<void> {
    await this.importFormatSelect.click({ force: true });
    const option = this.getActiveSelectOption(formatLabel);
    await expect(option).toBeVisible({ timeout: 10_000 });
    await option.click();
  }

  async openImportTask(): Promise<void> {
    await this.transferImportTask.click({ force: true });
    await expect(this.importTaskPanel).toBeVisible({ timeout: 10_000 });
    await expect(this.importTextarea).toBeVisible({ timeout: 10_000 });
  }

  async openExportTask(): Promise<void> {
    await this.transferExportTask.click({ force: true });
    await expect(this.exportTaskPanel).toBeVisible({ timeout: 10_000 });
  }

  async setReviewMode(mode: 'due' | 'cram'): Promise<void> {
    if (mode === 'cram') {
      await this.reviewCramModeOption.click({ force: true });
    } else {
      await this.reviewDueOnlyModeOption.click({ force: true });
    }
  }

  getReviewRatingButton(key: '1' | '2' | '3' | '4'): Locator {
    return this.page.locator(`[data-testid="flashcards-review-rate-${key}"]`);
  }

  async openManageFlashcardEdit(cardUuid: string): Promise<void> {
    const editButton = this.getManageFlashcardEditButton(cardUuid);
    await editButton.scrollIntoViewIfNeeded();
    await editButton.click();
    const drawerOpened = await this.editDrawer
      .waitFor({ state: 'visible', timeout: 1_000 })
      .then(() => true)
      .catch(() => false);
    if (drawerOpened) return;
    await this.getManageFlashcardRow(cardUuid).click();
    await this.page.keyboard.press('Enter');
    await expect(this.editDrawer).toBeVisible({ timeout: 10_000 });
  }

  // -- Tab Navigation --------------------------------------------------------

  async switchToTab(tab: 'study' | 'manage' | 'templates' | 'transfer' | 'scheduler'): Promise<void> {
    // Dismiss any overlays that might intercept clicks
    await dismissConnectionModals(this.page);
    const tabLocator = {
      study: this.studyTab,
      manage: this.manageTab,
      templates: this.templatesTab,
      scheduler: this.schedulerTab,
      transfer: this.transferTab,
    }[tab];
    await tabLocator.click({ force: true });
  }

  /** Returns true when the main tabs container is visible (server online + feature available) */
  async isOnline(): Promise<boolean> {
    return await this.tabsContainer.isVisible().catch(() => false);
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: 'Export flashcards button',
        locator: this.exportButton,
        expectation: {
          type: 'api_call',
          apiPattern: /\/api\/v1\/flashcards\/export/,
          method: 'GET',
        },
        setup: async () => {
          await this.switchToTab('transfer');
          await expect(this.importTextarea.or(this.exportDeckSelect)).toBeVisible({
            timeout: 5_000,
          });
        },
      },
      {
        name: 'Import flashcards button',
        locator: this.importButton,
        expectation: {
          type: 'api_call',
          apiPattern: /\/api\/v1\/flashcards\/import/,
          method: 'POST',
        },
        setup: async () => {
          await this.switchToTab('transfer');
          await expect(this.importTextarea.or(this.exportDeckSelect)).toBeVisible({
            timeout: 5_000,
          });
        },
      },
    ];
  }
}
