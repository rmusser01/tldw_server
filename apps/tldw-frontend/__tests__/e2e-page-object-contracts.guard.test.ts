import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const chatPagePath = path.resolve(testDir, '../e2e/utils/page-objects/ChatPage.ts');
const promptsWorkspacePagePath = path.resolve(
  testDir,
  '../e2e/utils/page-objects/PromptsWorkspacePage.ts'
);
const mediaPagePath = path.resolve(testDir, '../e2e/utils/page-objects/MediaPage.ts');
const notesPagePath = path.resolve(testDir, '../e2e/utils/page-objects/NotesPage.ts');
const worldBooksPagePath = path.resolve(testDir, '../e2e/utils/page-objects/WorldBooksPage.ts');
const writingPlaygroundPagePath = path.resolve(
  testDir,
  '../e2e/utils/page-objects/WritingPlaygroundPage.ts'
);
const audiobookStudioPagePath = path.resolve(
  testDir,
  '../e2e/utils/page-objects/AudiobookStudioPage.ts'
);
const knowledgeQaPagePath = path.resolve(testDir, '../e2e/utils/page-objects/KnowledgeQAPage.ts');
const agentRegistryPagePath = path.resolve(
  testDir,
  '../e2e/utils/page-objects/AgentRegistryPage.ts'
);
const agentTasksPagePath = path.resolve(testDir, '../e2e/utils/page-objects/AgentTasksPage.ts');
const journeyHelpersPath = path.resolve(testDir, '../e2e/utils/journey-helpers.ts');
const workflowFixturesPath = path.resolve(testDir, '../e2e/utils/fixtures.ts');
const notesFlashcardsJourneySpecPath = path.resolve(
  testDir,
  '../e2e/workflows/journeys/notes-flashcards.spec.ts'
);
const watchlistJourneySpecPath = path.resolve(
  testDir,
  '../e2e/workflows/journeys/watchlist-ingest-notify.spec.ts'
);
const watchlistInitialFeedPath = path.resolve(testDir, '../public/e2e/task2d-watchlist-feed.xml');
const watchlistUiRunFeedPath = path.resolve(
  testDir,
  '../public/e2e/task2d-watchlist-ui-run-feed.xml'
);

describe('e2e page object contracts', () => {
  it('keeps the chat workflow bound to the web chat surface contract', () => {
    const source = readFileSync(chatPagePath, 'utf8');

    expect(source).not.toContain('page.getByTestId("chat-header")');
    expect(source).not.toContain('/new saved chat/i');
    expect(source).not.toContain('getByText("General chat")');
    expect(source).not.toContain('getByRole("button", { name: /^General chat/i })');
    expect(source).toContain("article[aria-label*='Assistant message']");
    expect(source).toContain('getByRole("log", { name: /chat messages/i })');
    expect(source).toContain('Generating response');
    expect(source).toContain('assistantCount === 0');
    expect(source).toContain('getLastAssistantText');
    expect(source).toContain('getMessageBodyText');
    expect(source).toContain('Loading content');
    expect(source).toContain('Response complete');
    expect(source).toContain('structuredContentCount');
    expect(source).toContain('if (structuredContentCount > 0) return ""');
    expect(source).toContain('if (chipLabel.trim() && !/select a model/i.test(chipLabel))');
    expect(source).toContain('await selectModelTrigger.click()');
    expect(source).toContain('ensureModelSelected(force = false)');
    expect(source).toContain('if (!force && chipVisible)');
    expect(source).toContain('toHaveValue("", { timeout: 2_000 })');
    expect(source).toContain('await this.ensureModelSelected(true)');
    expect(source).toContain('name: /select a model/i');
  });

  it('waits for prompt persistence before closing the editor', () => {
    const source = readFileSync(promptsWorkspacePagePath, 'utf8');
    const createPromptSource = source.slice(
      source.indexOf('async createPrompt'),
      source.indexOf('async deletePrompt')
    );

    expect(createPromptSource).toContain('ant-notification-notice-success');
    expect(createPromptSource).toContain('Prompt Added');
    expect(createPromptSource).toContain('timeout: 30_000');
    expect(createPromptSource).toContain('toHaveCount(0, { timeout: 10_000 })');
    expect(createPromptSource).toContain('toHaveCount(1, { timeout: 30_000 })');
    expect(createPromptSource).toContain('await this.page.goto("/prompts"');
    expect(createPromptSource).toContain('await expect(this.fullPageEditor).toBeHidden');
    expect(createPromptSource).not.toContain('full-editor-back');
    expect(createPromptSource).not.toContain('fullPageEditorClosed || saveEnabled');
    expect(createPromptSource).not.toContain('expect(this.drawerSaveButton).toBeEnabled');
    expect(createPromptSource).not.toContain('existingSuccessCount');
  });

  it('keeps the media workflow bound to the media inspector shell contract', () => {
    const source = readFileSync(mediaPagePath, 'utf8');

    expect(source).not.toContain('waitForLoadState("networkidle"');
    expect(source).toContain('getByRole("heading", { name: /media inspector/i })');
    expect(source).toContain('getByTestId("media-results-list")');
  });

  it('keeps notes creation bound to the concrete save request contract', () => {
    const notesPageSource = readFileSync(notesPagePath, 'utf8');
    const journeyHelpersSource = readFileSync(journeyHelpersPath, 'utf8');
    const createNoteSource = notesPageSource.slice(
      notesPageSource.indexOf('async createNote'),
      notesPageSource.indexOf('async ensureMarkdownMode')
    );
    const assertNoteVisibleSource = notesPageSource.slice(
      notesPageSource.indexOf('async assertNoteVisible'),
      notesPageSource.indexOf('async assertNoteNotVisible')
    );

    expect(notesPageSource).toContain('expectApiCall(this.page');
    expect(notesPageSource).toContain('url: /\\/api\\/v1\\/notes\\/?$/');
    expect(notesPageSource).toContain('bodyContains');
    expect(notesPageSource).toContain('title: opts.title');
    expect(notesPageSource).toContain('content: opts.content');
    expect(notesPageSource).not.toContain('url: "/api/v1/notes"');
    expect(createNoteSource).not.toContain('await this.goto()');
    expect(assertNoteVisibleSource).not.toContain('await this.goto()');
    expect(assertNoteVisibleSource).not.toContain('await this.assertPageReady()');
    expect(journeyHelpersSource).toContain('const notesPage = new NotesPage(page)');
    expect(journeyHelpersSource).not.toContain('url: "/api/v1/notes"');
  });

  it('keeps world books interactions bound to the search-driven table contract', () => {
    const source = readFileSync(worldBooksPagePath, 'utf8');
    const worldBooksSpecPath = path.resolve(testDir, '../e2e/workflows/world-books.spec.ts');
    const worldBooksSpecSource = readFileSync(worldBooksSpecPath, 'utf8');

    expect(source).toContain('getByTestId("world-books-search-input")');
    expect(source).toContain('getByRole("row")');
    expect(source).toContain('manage entries');
    expect(source).toContain('quick attach characters');
    expect(source).toContain('getByRole("dialog", { name: title })');
    expect(source).toContain('getByRole("combobox", { name: /keywords/i })');
    expect(worldBooksSpecSource).toContain('characters\\/world-books');
  });

  it('keeps key page objects off direct networkidle readiness checks', () => {
    const sources = [
      readFileSync(writingPlaygroundPagePath, 'utf8'),
      readFileSync(audiobookStudioPagePath, 'utf8'),
      readFileSync(knowledgeQaPagePath, 'utf8'),
      readFileSync(worldBooksPagePath, 'utf8'),
      readFileSync(agentRegistryPagePath, 'utf8'),
      readFileSync(agentTasksPagePath, 'utf8'),
    ];

    for (const source of sources) {
      expect(source).not.toContain('waitForLoadState("networkidle"');
      expect(source).toContain('waitForAppShell');
    }
  });

  it('grants clipboard permissions for workflow tests', () => {
    const source = readFileSync(workflowFixturesPath, 'utf8');

    expect(source).toContain('grantPermissions(["clipboard-read", "clipboard-write"]');
    expect(source).toContain('new URL(TEST_CONFIG.webUrl).origin');
  });

  it('keeps the watchlist ingest journey bound to real run and notification contracts', () => {
    const source = readFileSync(watchlistJourneySpecPath, 'utf8');
    const initialFeed = readFileSync(watchlistInitialFeedPath, 'utf8');
    const uiRunFeed = readFileSync(watchlistUiRunFeedPath, 'utf8');

    expect(source).not.toContain('feature may not be implemented');
    expect(source).not.toContain('Watchlist page not available (404)');
    expect(source).not.toContain('Watchlist create button not found');
    expect(source).not.toContain('watchlists/jobs/300/run');
    expect(source).not.toMatch(/watchlists\\*\/jobs\\*\/300\\*\/run/);
    expect(source).toContain('new RegExp(`/api/v1/watchlists/jobs/${jobId}/run$`)');
    expect(source).not.toContain('getByRole("button", { name: "Open Monitors" })');
    expect(source).not.toContain('getByRole("button", { name: "Open Activity" })');
    expect(source).not.toContain('watchlists-item-row-9001');
    expect(source).toContain('newWatchlistJourneyState');
    expect(source).toContain('provisionPersistedWatchlistJourneyState');
    expect(source).toContain('updateSourceForUiRun');
    expect(source).toContain('waitForOwnedRunItem');
    expect(source).toContain('/api/v1/watchlists/items?run_id=${runId}');
    expect(source).toContain('/api/v1/watchlists/sources/${source.id}/seen');
    expect(source).toContain('/api/v1/watchlists/sources/${state.sourceId}/seen');
    expect(source).toContain(
      'settings: { rss: { use_feed_content_if_available: true, feed_content_min_chars: 1 } }'
    );
    expect(source).toContain('cleanupFailures');
    expect(source).toContain('watchlists-secondary-activity');
    expect(source).toContain('watchlists-run-open-outputs-${completedRunId}');
    expect(source).toContain('NotificationsPage');
    expect(initialFeed).toContain('<guid isPermaLink="false">task2d-persisted-briefing-v1</guid>');
    expect(uiRunFeed).toContain('<guid isPermaLink="false">task2d-ui-run-now-v1</guid>');
    expect(initialFeed).toContain('https://example.test/task2d/persisted-briefing-v1');
    expect(uiRunFeed).toContain('https://example.test/task2d/ui-run-now-v1');
    expect(initialFeed).not.toContain('127.0.0.1:62462');
    expect(uiRunFeed).not.toContain('127.0.0.1:62462');
    expect(initialFeed).not.toContain('task2d-ui-run-now-v1');
    expect(uiRunFeed).not.toContain('task2d-persisted-briefing-v1');
  });

  it('keeps the notes to flashcards journey aligned with the partial-save transfer contract', () => {
    const source = readFileSync(notesFlashcardsJourneySpecPath, 'utf8');

    expect(source).toContain(
      'page.getByText(/Saved \\d+ (?:generated )?cards(?:; \\d+ failed\\.)?/i)'
    );
    expect(source).toContain('toBeGreaterThan(initialCardCount)');
    expect(source).not.toContain('Saved \\\\d+ generated cards/i');
    expect(source).not.toContain('manageTopBar');
  });
});
