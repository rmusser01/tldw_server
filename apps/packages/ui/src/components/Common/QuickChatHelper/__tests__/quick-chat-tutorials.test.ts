import { describe, expect, it } from "vitest"
import { buildQuickChatPageTutorialEntries } from "../QuickChatGuidesPanel"

describe("quick chat per-page tutorials", () => {
  it.each([
    "/chat-workspace",
    "chrome-extension://example/options.html#/chat-workspace"
  ])("offers the workspace chat tour from %s", (route) => {
    const entries = buildQuickChatPageTutorialEntries(route, [])

    expect(entries).toEqual([
      expect.objectContaining({
        tutorial: expect.objectContaining({ id: "chat-workspace-basics" }),
        isLocked: false
      })
    ])
  })

  it("returns page tutorials for canonical /chat route", () => {
    const entries = buildQuickChatPageTutorialEntries("/chat", [])

    expect(entries.length).toBeGreaterThan(0)
    expect(entries.some((entry) => entry.tutorial.id === "playground-basics")).toBe(
      true
    )
  })

  it("normalizes legacy route aliases", () => {
    const entries = buildQuickChatPageTutorialEntries("/options/playground", [])

    expect(entries.some((entry) => entry.tutorial.id === "playground-basics")).toBe(
      true
    )
  })

  it("returns media page tutorials for /media route", () => {
    const entries = buildQuickChatPageTutorialEntries("/media", [])

    expect(entries.some((entry) => entry.tutorial.id === "media-basics")).toBe(true)
  })

  it("maps /options/media alias to media tutorials", () => {
    const entries = buildQuickChatPageTutorialEntries("/options/media", [])

    expect(entries.some((entry) => entry.tutorial.id === "media-basics")).toBe(true)
  })

  it("returns prompts tutorials for /prompts route", () => {
    const entries = buildQuickChatPageTutorialEntries("/prompts", [])

    expect(entries.some((entry) => entry.tutorial.id === "prompts-basics")).toBe(
      true
    )
  })

  it("returns world books tutorials for /world-books route", () => {
    const entries = buildQuickChatPageTutorialEntries("/world-books", [])

    expect(
      entries.some((entry) => entry.tutorial.id === "world-books-basics")
    ).toBe(true)
  })

  it("returns evaluations tutorials for /evaluations route", () => {
    const entries = buildQuickChatPageTutorialEntries("/evaluations", [])

    expect(
      entries.some((entry) => entry.tutorial.id === "evaluations-basics")
    ).toBe(true)
  })

  it("returns notes tutorials for /notes route", () => {
    const entries = buildQuickChatPageTutorialEntries("/notes", [])

    expect(entries.some((entry) => entry.tutorial.id === "notes-basics")).toBe(
      true
    )
  })

  it("returns flashcards tutorials for /flashcards route", () => {
    const entries = buildQuickChatPageTutorialEntries("/flashcards", [])

    expect(
      entries.some((entry) => entry.tutorial.id === "flashcards-basics")
    ).toBe(true)
  })

  it("maps /options/evaluations alias to evaluations tutorials", () => {
    const entries = buildQuickChatPageTutorialEntries("/options/evaluations", [])

    expect(
      entries.some((entry) => entry.tutorial.id === "evaluations-basics")
    ).toBe(true)
  })

  it("marks advanced tutorials locked until prerequisites are complete", () => {
    const entries = buildQuickChatPageTutorialEntries("/chat", [])
    const toolsTutorial = entries.find(
      (entry) => entry.tutorial.id === "playground-tools"
    )

    expect(toolsTutorial?.isLocked).toBe(true)

    const unlockedEntries = buildQuickChatPageTutorialEntries("/chat", [
      "playground-basics"
    ])
    const unlockedToolsTutorial = unlockedEntries.find(
      (entry) => entry.tutorial.id === "playground-tools"
    )

    expect(unlockedToolsTutorial?.isLocked).toBe(false)
  })

  it("returns empty tutorial list for routes without definitions", () => {
    const entries = buildQuickChatPageTutorialEntries("/settings/health", [])
    expect(entries).toEqual([])
  })

  it("resolves knowledge tutorials for knowledge subroutes", () => {
    const entries = buildQuickChatPageTutorialEntries("/knowledge/thread/abc123", [])
    expect(entries.some((entry) => entry.tutorial.id === "knowledge-basics")).toBe(
      true
    )
  })

  it("still resolves tutorials when runtime path is sidepanel", () => {
    const originalPath = `${window.location.pathname}${window.location.search}${window.location.hash}`

    try {
      window.history.replaceState({}, "", "/sidepanel.html")
      const entries = buildQuickChatPageTutorialEntries("/chat", [])
      expect(entries.some((entry) => entry.tutorial.id === "playground-basics")).toBe(
        true
      )
    } finally {
      window.history.replaceState({}, "", originalPath || "/")
    }
  })
})
