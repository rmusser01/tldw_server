import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const chatPagePath = path.resolve(testDir, "../e2e/utils/page-objects/ChatPage.ts")
const mediaPagePath = path.resolve(testDir, "../e2e/utils/page-objects/MediaPage.ts")
const notesPagePath = path.resolve(testDir, "../e2e/utils/page-objects/NotesPage.ts")
const worldBooksPagePath = path.resolve(testDir, "../e2e/utils/page-objects/WorldBooksPage.ts")
const writingPlaygroundPagePath = path.resolve(
  testDir,
  "../e2e/utils/page-objects/WritingPlaygroundPage.ts"
)
const audiobookStudioPagePath = path.resolve(
  testDir,
  "../e2e/utils/page-objects/AudiobookStudioPage.ts"
)
const audioStudioPagePath = path.resolve(testDir, "../e2e/utils/page-objects/AudioStudioPage.ts")
const knowledgeQaPagePath = path.resolve(
  testDir,
  "../e2e/utils/page-objects/KnowledgeQAPage.ts"
)
const agentRegistryPagePath = path.resolve(
  testDir,
  "../e2e/utils/page-objects/AgentRegistryPage.ts"
)
const agentTasksPagePath = path.resolve(
  testDir,
  "../e2e/utils/page-objects/AgentTasksPage.ts"
)
const journeyHelpersPath = path.resolve(testDir, "../e2e/utils/journey-helpers.ts")
const workflowFixturesPath = path.resolve(testDir, "../e2e/utils/fixtures.ts")
const notesFlashcardsJourneySpecPath = path.resolve(
  testDir,
  "../e2e/workflows/journeys/notes-flashcards.spec.ts"
)
const watchlistJourneySpecPath = path.resolve(
  testDir,
  "../e2e/workflows/journeys/watchlist-ingest-notify.spec.ts"
)

describe("e2e page object contracts", () => {
  it("keeps the chat workflow bound to the web chat surface contract", () => {
    const source = readFileSync(chatPagePath, "utf8")

    expect(source).not.toContain('page.getByTestId("chat-header")')
    expect(source).not.toContain('/new saved chat/i')
    expect(source).not.toContain('getByText("General chat")')
    expect(source).not.toContain('getByRole("button", { name: /^General chat/i })')
    expect(source).toContain("article[aria-label*='Assistant message']")
    expect(source).toContain('getByRole("log", { name: /chat messages/i })')
    expect(source).toContain("Generating response")
    expect(source).toContain("assistantCount === 0")
    expect(source).toContain("getLastAssistantText")
    expect(source).toContain("getMessageBodyText")
    expect(source).toContain("Loading content")
    expect(source).toContain("Response complete")
    expect(source).toContain("name: /select a model/i")
  })

  it("keeps the media workflow bound to the media inspector shell contract", () => {
    const source = readFileSync(mediaPagePath, "utf8")

    expect(source).not.toContain('waitForLoadState("networkidle"')
    expect(source).toContain('getByRole("heading", { name: /media inspector/i })')
    expect(source).toContain('getByTestId("media-results-list")')
  })

  it("keeps notes creation bound to the concrete save request contract", () => {
    const notesPageSource = readFileSync(notesPagePath, "utf8")
    const journeyHelpersSource = readFileSync(journeyHelpersPath, "utf8")

    expect(notesPageSource).toContain("expectApiCall(this.page")
    expect(notesPageSource).toContain("url: /\\/api\\/v1\\/notes\\/?$/")
    expect(notesPageSource).toContain("bodyContains")
    expect(notesPageSource).toContain("title: opts.title")
    expect(notesPageSource).toContain("content: opts.content")
    expect(notesPageSource).not.toContain('url: "/api/v1/notes"')
    expect(journeyHelpersSource).toContain("const notesPage = new NotesPage(page)")
    expect(journeyHelpersSource).not.toContain('url: "/api/v1/notes"')
  })

  it("keeps world books interactions bound to the search-driven table contract", () => {
    const source = readFileSync(worldBooksPagePath, "utf8")
    const worldBooksSpecPath = path.resolve(testDir, "../e2e/workflows/world-books.spec.ts")
    const worldBooksSpecSource = readFileSync(worldBooksSpecPath, "utf8")

    expect(source).toContain('getByTestId("world-books-search-input")')
    expect(source).toContain('getByRole("row")')
    expect(source).toContain("manage entries")
    expect(source).toContain("quick attach characters")
    expect(source).toContain('getByRole("dialog", { name: title })')
    expect(source).toContain('getByRole("combobox", { name: /keywords/i })')
    expect(worldBooksSpecSource).toContain("characters\\/world-books")
  })

  it("keeps key page objects off direct networkidle readiness checks", () => {
    const sources = [
      readFileSync(writingPlaygroundPagePath, "utf8"),
      readFileSync(audioStudioPagePath, "utf8"),
      readFileSync(knowledgeQaPagePath, "utf8"),
      readFileSync(worldBooksPagePath, "utf8"),
      readFileSync(agentRegistryPagePath, "utf8"),
      readFileSync(agentTasksPagePath, "utf8"),
    ]

    for (const source of sources) {
      expect(source).not.toContain('waitForLoadState("networkidle"')
      expect(source).toContain("waitForAppShell")
    }

    // The legacy route inherits readiness from the canonical Audio Studio page.
    const compatibilitySource = readFileSync(audiobookStudioPagePath, "utf8")
    expect(compatibilitySource).toContain('import { AudioStudioPage } from "./AudioStudioPage"')
    expect(compatibilitySource).toContain("extends AudioStudioPage")
    expect(compatibilitySource).toContain("await this.gotoCompatibilityRoute()")
    expect(compatibilitySource).not.toContain("async assertPageReady(")
    expect(compatibilitySource).not.toContain('waitForLoadState("networkidle"')
  })

  it("grants clipboard permissions for workflow tests", () => {
    const source = readFileSync(workflowFixturesPath, "utf8")

    expect(source).toContain('grantPermissions(["clipboard-read", "clipboard-write"]')
    expect(source).toContain("new URL(TEST_CONFIG.webUrl).origin")
  })

  it("keeps the watchlist ingest journey bound to real run and notification contracts", () => {
    const source = readFileSync(watchlistJourneySpecPath, "utf8")

    expect(source).not.toContain("feature may not be implemented")
    expect(source).not.toContain("Watchlist page not available (404)")
    expect(source).not.toContain("Watchlist create button not found")
    expect(source).toMatch(/watchlists\\\/jobs\\\/300\\\/run/)
    expect(source).toContain('getByTestId("watchlists-open-command-palette").click()')
    expect(source).toContain('getByTestId("watchlists-command-nav-monitors").click()')
    expect(source).toContain('expect(page.getByLabel(/Monitors table/i)).toBeVisible()')
    expect(source).toContain('getByRole("button", { name: "Open Activity" })')
    expect(source).toContain("watchlists-secondary-activity")
    expect(source).toContain("watchlists-item-row-9001")
    expect(source).toContain("NotificationsPage")
  })

  it("requires exact supported saves and source lineage in the notes to flashcards journey", () => {
    const source = readFileSync(notesFlashcardsJourneySpecPath, "utf8")
    expect(source).toContain("assertBiologyCardSet(generated.flashcards)")
    expect(source).toContain("expect(saves).toHaveLength(5)")
    expect(source).toContain("expect(new Set(savedCards.map((card) => card.uuid)).size).toBe(5)")
    expect(source).toContain("expect(stored.items).toHaveLength(5)")
    expect(source).toContain("source_ref_id: noteId")
    expect(source).toContain("rating: 5")
    expect(source).not.toContain("toBeGreaterThan(initialCardCount)")
    expect(source).not.toContain("manageTopBar")
  })
})
