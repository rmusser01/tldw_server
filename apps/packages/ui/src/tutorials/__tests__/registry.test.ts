import { afterEach, describe, expect, it } from "vitest"
import {
  TUTORIAL_REGISTRY,
  areTutorialPrerequisitesMet,
  getPrimaryTutorialForRoute,
  getNextTutorialInSequence,
  getTutorialsForRoute,
  isTutorialRuntimeSuppressed,
  normalizeTutorialRoute,
  type TutorialDefinition
} from "../registry"

const injectedTutorials: TutorialDefinition[] = []

afterEach(() => {
  while (injectedTutorials.length > 0) {
    const injected = injectedTutorials.pop()
    if (!injected) continue
    const index = TUTORIAL_REGISTRY.findIndex((tutorial) => tutorial.id === injected.id)
    if (index >= 0) {
      TUTORIAL_REGISTRY.splice(index, 1)
    }
  }
})

describe("tutorial registry route matching", () => {
  it("matches playground tutorials on canonical /chat route", () => {
    const tutorials = getTutorialsForRoute("/chat")

    expect(tutorials.length).toBeGreaterThan(0)
    expect(tutorials.some((tutorial) => tutorial.id === "playground-basics")).toBe(
      true
    )
  })

  it("matches legacy /options/playground alias to /chat tutorials", () => {
    const tutorials = getTutorialsForRoute("/options/playground")

    expect(tutorials.length).toBeGreaterThan(0)
    expect(tutorials.some((tutorial) => tutorial.id === "playground-basics")).toBe(
      true
    )
  })

  it("normalizes extension hash urls for tutorial lookup", () => {
    const tutorials = getTutorialsForRoute(
      "chrome-extension://abc/options.html#/chat?tab=casual"
    )

    expect(tutorials.some((tutorial) => tutorial.id === "playground-basics")).toBe(
      true
    )
  })

  it("maps knowledge thread routes to canonical knowledge tutorials", () => {
    const tutorials = getTutorialsForRoute("/knowledge/thread/abc123")

    expect(tutorials.some((tutorial) => tutorial.id === "knowledge-basics")).toBe(
      true
    )
  })

  it("maps knowledge shared routes to canonical knowledge tutorials", () => {
    const tutorials = getTutorialsForRoute("/knowledge/shared/share-token")

    expect(tutorials.some((tutorial) => tutorial.id === "knowledge-basics")).toBe(
      true
    )
  })

  it("supports wildcard route patterns", () => {
    const wildcardTutorial: TutorialDefinition = {
      id: "test-settings-wildcard",
      routePattern: "/settings/*",
      labelKey: "tutorials:test.settings.label",
      labelFallback: "Settings wildcard",
      descriptionKey: "tutorials:test.settings.description",
      descriptionFallback: "Wildcard tutorial",
      steps: [
        {
          target: "body",
          titleKey: "tutorials:test.settings.stepTitle",
          titleFallback: "Step",
          contentKey: "tutorials:test.settings.stepContent",
          contentFallback: "Wildcard match test"
        }
      ]
    }

    TUTORIAL_REGISTRY.push(wildcardTutorial)
    injectedTutorials.push(wildcardTutorial)

    const tutorials = getTutorialsForRoute("/settings/health")
    expect(tutorials.some((tutorial) => tutorial.id === wildcardTutorial.id)).toBe(
      true
    )
  })

  it("returns the basics tutorial as the primary tutorial for /chat", () => {
    const primaryTutorial = getPrimaryTutorialForRoute("/chat")

    expect(primaryTutorial?.id).toBe("playground-basics")
  })

  it("filters prerequisite-gated tutorials until their prerequisites are completed", () => {
    const lockedTutorial: TutorialDefinition = {
      id: "test-locked-knowledge-tour",
      routePattern: "/knowledge",
      labelKey: "tutorials:test.lockedKnowledge.label",
      labelFallback: "Locked Knowledge Tour",
      descriptionKey: "tutorials:test.lockedKnowledge.description",
      descriptionFallback: "A locked route tutorial",
      prerequisites: ["getting-started"],
      priority: 0,
      steps: [
        {
          target: "body",
          titleKey: "tutorials:test.lockedKnowledge.stepTitle",
          titleFallback: "Step",
          contentKey: "tutorials:test.lockedKnowledge.stepContent",
          contentFallback: "Locked route tutorial test"
        }
      ]
    }

    TUTORIAL_REGISTRY.push(lockedTutorial)
    injectedTutorials.push(lockedTutorial)

    expect(areTutorialPrerequisitesMet(lockedTutorial, [])).toBe(false)
    expect(
      areTutorialPrerequisitesMet(lockedTutorial, new Set(["getting-started"]))
    ).toBe(true)
    expect(
      getTutorialsForRoute("/knowledge", { completedTutorialIds: [] }).some(
        (tutorial) => tutorial.id === lockedTutorial.id
      )
    ).toBe(false)
    expect(
      getTutorialsForRoute("/knowledge", {
        completedTutorialIds: new Set<string>(),
        includeLocked: true
      }).some((tutorial) => tutorial.id === lockedTutorial.id)
    ).toBe(true)
    expect(
      getTutorialsForRoute("/knowledge", {
        completedTutorialIds: ["getting-started"]
      }).some((tutorial) => tutorial.id === lockedTutorial.id)
    ).toBe(true)
  })

  it("uses knowledge basics as the second getting started sequence step", () => {
    expect(
      TUTORIAL_REGISTRY.some(
        (tutorial) => tutorial.id === "getting-started-knowledge"
      )
    ).toBe(false)

    const primaryTutorial = getPrimaryTutorialForRoute("/knowledge", {
      completedTutorialIds: ["getting-started"]
    })

    expect(primaryTutorial?.id).toBe("knowledge-basics")
  })

  it("resolves the next eligible tutorial in the getting started sequence", () => {
    expect(getNextTutorialInSequence("getting-started")?.id).toBe(
      "knowledge-basics"
    )
    expect(
      getNextTutorialInSequence("knowledge-basics", new Set(["getting-started"]))
        ?.id
    ).toBe("document-workspace-basics")
  })

  it("starts the document workspace tutorial on an always-rendered route shell", () => {
    const primaryTutorial = getPrimaryTutorialForRoute("/document-workspace")

    expect(primaryTutorial?.id).toBe("document-workspace-basics")
    expect(primaryTutorial?.steps[0]?.target).toBe(
      '[data-testid="document-workspace-root"]'
    )
    expect(
      primaryTutorial?.steps.some(
        (step) => step.target === '[data-testid="document-open-picker-button"]'
      )
    ).toBe(true)
    const lastStep = primaryTutorial?.steps[primaryTutorial.steps.length - 1]
    expect(lastStep?.target).toBe(
      '[data-testid="document-navigation"]'
    )
  })

  it("includes basics tutorials for all P0/P1 page routes", () => {
    const expectedBasicsByRoute: Record<string, string> = {
      "/chat": "playground-basics",
      "/research-workspace": "research-workspace-basics",
      "/media": "media-basics",
      "/knowledge": "knowledge-basics",
      "/characters": "characters-basics",
      "/prompts": "prompts-basics",
      "/evaluations": "evaluations-basics",
      "/notes": "notes-basics",
      "/flashcards": "flashcards-basics",
      "/world-books": "world-books-basics",
      "/document-workspace": "document-workspace-basics"
    }

    for (const [route, expectedId] of Object.entries(expectedBasicsByRoute)) {
      const tutorials = getTutorialsForRoute(route)
      expect(tutorials.some((tutorial) => tutorial.id === expectedId)).toBe(true)
    }
  })

  it("normalizes legacy paths to canonical routes", () => {
    expect(normalizeTutorialRoute("/options/playground")).toBe("/chat")
    expect(normalizeTutorialRoute("#/research-workspace?tab=chat")).toBe(
      "/research-workspace"
    )
    expect(normalizeTutorialRoute("/options/media")).toBe("/media")
    expect(normalizeTutorialRoute("/options/knowledge")).toBe("/knowledge")
    expect(normalizeTutorialRoute("/options/characters")).toBe("/characters")
    expect(normalizeTutorialRoute("/options/prompts")).toBe("/prompts")
    expect(normalizeTutorialRoute("/options/evaluations")).toBe("/evaluations")
    expect(normalizeTutorialRoute("/options/notes")).toBe("/notes")
    expect(normalizeTutorialRoute("/options/flashcards")).toBe("/flashcards")
    expect(normalizeTutorialRoute("/options/world-books")).toBe("/world-books")
    expect(normalizeTutorialRoute("/options/document-workspace")).toBe(
      "/document-workspace"
    )
    expect(normalizeTutorialRoute("/knowledge/thread/abc123")).toBe("/knowledge")
    expect(normalizeTutorialRoute("/knowledge/shared/share-token")).toBe("/knowledge")
  })

  it("suppresses tutorials in sidepanel runtime context", () => {
    const originalPath = `${window.location.pathname}${window.location.search}${window.location.hash}`

    try {
      window.history.replaceState({}, "", "/sidepanel.html")
      expect(isTutorialRuntimeSuppressed()).toBe(true)
      expect(getTutorialsForRoute("/chat")).toEqual([])
    } finally {
      window.history.replaceState({}, "", originalPath || "/")
    }
  })

  it("can bypass sidepanel suppression when explicitly requested", () => {
    const originalPath = `${window.location.pathname}${window.location.search}${window.location.hash}`

    try {
      window.history.replaceState({}, "", "/sidepanel.html")
      const tutorials = getTutorialsForRoute("/chat", {
        ignoreRuntimeSuppression: true
      })
      expect(tutorials.some((tutorial) => tutorial.id === "playground-basics")).toBe(
        true
      )
    } finally {
      window.history.replaceState({}, "", originalPath || "/")
    }
  })
})
