/**
 * Page Object for the Flashcards workspace
 *
 * The route renders FlashcardsWorkspace which shows either:
 * - A connection/offline banner when the server is unreachable
 * - FlashcardsManager with tabs for Study, Manage, Import / Export, Templates, and Scheduler
 *
 * API base paths:
 *   /api/v1/flashcards        (cards CRUD, review, generate, import, export)
 *   /api/v1/flashcards/decks  (deck CRUD)
 */
import { type Page, type Locator, expect } from '@playwright/test';
import { BasePage, type InteractiveElement } from './BasePage';
import { waitForAppShell, waitForConnection, dismissConnectionModals } from '../helpers';

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

  get templatesCreateButton(): Locator {
    return this.page.getByRole('button', { name: /create template/i });
  }

  get templatesErrorAlert(): Locator {
    return this.page.getByRole('alert').filter({ hasText: /could not load templates/i });
  }

  get transferTab(): Locator {
    return this.page.getByRole('tab', { name: /import\s*\/\s*export/i });
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

  get reviewModeToggle(): Locator {
    return this.page.locator('[data-testid="flashcards-review-mode-toggle"]');
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

  get reviewEmptyCard(): Locator {
    return this.page.locator('[data-testid="flashcards-review-empty-card"]');
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
    return this.page.locator('.ant-drawer-content').filter({ hasText: 'Create Flashcard' }).last();
  }

  get createDrawerDeckSelect(): Locator {
    return this.createDrawer.locator('.ant-select').first();
  }

  get editDrawer(): Locator {
    return this.page.locator('.ant-drawer-content').filter({ hasText: 'Edit Flashcard' }).last();
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

  get importTextarea(): Locator {
    return this.page.locator('[data-testid="flashcards-import-textarea"]');
  }

  get importButton(): Locator {
    return this.page.locator('[data-testid="flashcards-import-button"]');
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
    return this.page
      .locator('.ant-select-dropdown')
      .filter({ has: this.page.getByRole('option', { name: optionName, exact }) })
      .locator('[role="option"]')
      .filter({ hasText: optionName })
      .first();
  }

  async selectManageDeckByName(deckName: string): Promise<void> {
    await this.manageDeckSelect.click({ force: true });
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
    await this.createDrawerDeckSelect.click({ force: true });
    const deckOption = this.getActiveSelectOption(deckName);
    await expect(deckOption).toBeVisible({ timeout: 10_000 });
    await deckOption.click();
  }

  async openManageFlashcardEdit(cardUuid: string): Promise<void> {
    await this.getManageFlashcardEditButton(cardUuid).click({ force: true });
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
