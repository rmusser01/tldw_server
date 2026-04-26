import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { getTutorialById } from "../registry"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../")

const readSource = (relativePath: string): string =>
  readFileSync(path.resolve(srcRoot, relativePath), "utf8")

describe("P0/P1 tutorial selector contracts", () => {
  it("uses stable selector formats for all P0/P1 tutorial steps", () => {
    const p0p1TutorialIds = [
      "playground-basics",
      "playground-tools",
      "playground-voice",
      "workspace-playground-basics",
      "media-basics",
      "knowledge-basics",
      "characters-basics",
      "prompts-basics",
      "evaluations-basics",
      "notes-basics",
      "flashcards-basics",
      "world-books-basics"
    ]

    for (const tutorialId of p0p1TutorialIds) {
      const tutorial = getTutorialById(tutorialId)
      expect(tutorial).toBeDefined()
      if (!tutorial) continue

      for (const step of tutorial.steps) {
        const selectors = step.target
          .split(",")
          .map((value) => value.trim())
          .filter((value) => value.length > 0)

        expect(selectors.length).toBeGreaterThan(0)
        for (const selector of selectors) {
          const isStableSelector =
            selector.startsWith('[data-testid="') || selector.startsWith("#")
          if (!isStableSelector) {
            throw new Error(`Unstable selector: ${selector}`)
          }
          expect(isStableSelector).toBe(true)
        }
      }
    }
  })

  it("keeps required playground tutorial anchors in source", () => {
    const formContent = readSource("components/Option/Playground/PlaygroundForm.tsx")
    const toolbarContent = readSource("components/Option/Playground/ComposerToolbar.tsx")
    const textareaContent = readSource("components/Option/Playground/ComposerTextarea.tsx")
    const promptSelectContent = readSource("components/Common/PromptSelect.tsx")
    const headerContent = readSource("components/Layouts/ChatHeader.tsx")
    const chatSidebarContent = readSource("components/Common/ChatSidebar.tsx")

    const formAnchors = [
      'data-testid="model-selector"',
      'data-testid="tools-button"',
      'data-testid="attachment-button"',
      'data-testid="mcp-tools-toggle"',
      'data-testid="voice-chat-button"'
    ]
    for (const anchor of formAnchors) {
      expect(formContent).toContain(anchor)
    }

    const toolbarAnchors = [
      'data-testid="knowledge-search-toggle"',
      'data-testid="web-search-toggle"',
      'data-testid="dictation-button"'
    ]
    for (const anchor of toolbarAnchors) {
      expect(toolbarContent).toContain(anchor)
    }

    expect(textareaContent).toContain('data-testid="chat-input"')
    expect(promptSelectContent).toContain('dataTestId="chat-prompt-select"')
    expect(headerContent).toContain('data-testid="new-chat-button"')
    expect(chatSidebarContent).toContain('data-testid="chat-sidebar-new-chat"')
  })

  it("keeps required workspace playground tutorial anchors in source", () => {
    const workspaceContent = readSource("components/Option/WorkspacePlayground/index.tsx")
    const headerContent = readSource("components/Option/WorkspacePlayground/WorkspaceHeader.tsx")

    expect(workspaceContent).toContain('id="workspace-sources-panel"')
    expect(workspaceContent).toContain('id="workspace-main-content"')
    expect(workspaceContent).toContain('id="workspace-studio-panel"')
    expect(headerContent).toContain('data-testid="workspace-workspaces-button"')
  })

  it("keeps required media tutorial anchors in source", () => {
    const searchContent = readSource("components/Media/SearchBar.tsx")
    const reviewContent = readSource("components/Review/ViewMediaPage.tsx")
    const resultsContent = readSource("components/Media/ResultsList.tsx")
    const viewerContent = readSource("components/Media/ContentViewer.tsx")

    expect(searchContent).toContain('data-testid="media-search-input"')
    expect(reviewContent).toContain('id="media-search-panel"')
    expect(reviewContent).toContain('data-testid="media-search-submit"')
    expect(resultsContent).toContain('data-testid="media-results-list"')
    expect(viewerContent).toContain('data-testid="content-scroll-container"')
    expect(viewerContent).toContain('data-testid="content-viewer-empty"')
    expect(reviewContent).toContain('data-testid="media-library-tools-toggle"')
  })

  it("keeps required knowledge tutorial anchors in source", () => {
    const searchContent = readSource("components/Option/KnowledgeQA/SearchBar.tsx")
    const contextBarContent = readSource(
      "components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx"
    )
    const layoutContent = readSource("components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx")
    const historyContent = readSource("components/Option/KnowledgeQA/HistorySidebar.tsx")

    expect(searchContent).toContain('id="knowledge-search-input"')
    expect(contextBarContent).toContain('id="knowledge-source-selector-toggle"')
    expect(layoutContent).toContain('data-testid="knowledge-search-shell"')
    expect(layoutContent).toContain('data-testid="knowledge-results-shell"')
    expect(historyContent).toContain('data-testid="knowledge-history-desktop-open"')
    expect(historyContent).toContain('data-testid="knowledge-history-desktop-collapsed"')
    expect(historyContent).toContain('data-testid="knowledge-history-mobile-open"')
  })

  it("keeps required characters tutorial anchors in source", () => {
    const managerContent = readSource("components/Option/Characters/Manager.tsx")
    const anchors = [
      'data-testid="characters-new-button"',
      'data-testid="characters-search-input"',
      'data-testid="characters-scope-segmented"',
      'data-testid="characters-view-mode-segmented"',
      'data-testid="characters-table-view"',
      'data-testid="characters-gallery-view"'
    ]

    for (const anchor of anchors) {
      expect(managerContent).toContain(anchor)
    }
  })

  it("keeps required prompts tutorial anchors in source", () => {
    const content = readSource("components/Option/Prompt/index.tsx")
    const anchors = [
      'data-testid="prompts-segmented"',
      'data-testid="prompts-add"',
      'data-testid="prompts-search"',
      'data-testid="prompts-type-filter"',
      'data-testid="prompts-tag-filter"',
      'data-testid="prompts-export"',
      'data-testid="prompts-import"'
    ]

    for (const anchor of anchors) {
      expect(content).toContain(anchor)
    }
  })

  it("keeps required evaluations tutorial anchors in source", () => {
    const pageContent = readSource("components/Option/Evaluations/EvaluationsPage.tsx")
    const tabContent = readSource(
      "components/Option/Evaluations/tabs/EvaluationsTab.tsx"
    )
    const anchors = [
      'data-testid="evaluations-page-title"',
      'data-testid="evaluations-tabs"',
      'data-testid="evaluations-create-button"',
      'data-testid="evaluations-list-card"',
      'data-testid="evaluations-detail-card"'
    ]

    expect(pageContent).toContain(anchors[0])
    expect(pageContent).toContain(anchors[1])
    expect(tabContent).toContain(anchors[2])
    expect(tabContent).toContain(anchors[3])
    expect(tabContent).toContain(anchors[4])
  })

  it("keeps required notes tutorial anchors in source", () => {
    const headerContent = readSource("components/Notes/NotesEditorHeader.tsx")
    const editorPaneContent = readSource("components/Notes/NotesEditorPane.tsx")
    const sidebarContent = readSource("components/Notes/NotesSidebar.tsx")

    // Sidebar anchors
    const sidebarAnchors = [
      'data-testid="notes-list-region"',
      'data-testid="notes-mode-active"',
      'data-testid="notes-mode-trash"',
      'data-testid="notes-sort-select"',
      'data-testid="notes-notebook-select"'
    ]
    for (const anchor of sidebarAnchors) {
      expect(sidebarContent).toContain(anchor)
    }

    // Editor pane anchors
    expect(editorPaneContent).toContain('data-testid="notes-editor-region"')
    expect(editorPaneContent).toContain('data-testid="notes-keywords-editor"')
    expect(editorPaneContent).toContain('testId="notes-section-connections"')

    // Header anchors
    expect(headerContent).toContain('data-testid="notes-save-button"')

    // Organize section anchor
    expect(sidebarContent).toContain('testId="notes-section-organize"')
  })

  it("keeps required flashcards tutorial anchors in source", () => {
    const managerContent = readSource("components/Flashcards/FlashcardsManager.tsx")
    const reviewContent = readSource("components/Flashcards/tabs/ReviewTab.tsx")
    const manageContent = readSource("components/Flashcards/tabs/ManageTab.tsx")
    const importExportContent = readSource(
      "components/Flashcards/tabs/ImportExportTab.tsx"
    )

    expect(managerContent).toContain('data-testid="flashcards-tabs"')
    expect(managerContent).toContain('data-testid="flashcards-to-quiz-cta"')
    expect(reviewContent).toContain('data-testid="flashcards-review-topbar"')
    expect(reviewContent).toContain('data-testid="flashcards-review-empty-card"')
    expect(manageContent).toContain('data-testid="flashcards-manage-topbar"')
    expect(manageContent).toContain('data-testid="flashcards-manage-search"')
    expect(importExportContent).toContain('data-testid="flashcards-import-format"')
    expect(importExportContent).toContain(
      'data-testid="flashcards-import-help-accordion"'
    )
  })

  it("keeps required world books tutorial anchors in source", () => {
    const workspaceContent = readSource(
      "components/Option/WorldBooks/WorldBooksWorkspace.tsx"
    )
    const managerContent = readSource("components/Option/WorldBooks/Manager.tsx")

    expect(workspaceContent).toContain('data-testid="world-books-tutorial-shell"')
    expect(managerContent).toContain('data-testid="world-books-search-input"')
    expect(managerContent).toContain('data-testid="world-books-enabled-filter"')
    expect(managerContent).toContain('data-testid="world-books-attachment-filter"')
    expect(managerContent).toContain('data-testid="world-books-table"')
    expect(managerContent).toContain('data-testid="world-books-new-button"')
    expect(managerContent).toContain('data-testid="world-books-import-button"')
  })
})
