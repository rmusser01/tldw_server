// @vitest-environment jsdom
import { describe, expect, it, beforeEach, vi } from "vitest"

/**
 * Mirrors the condition used in QuizPlayground.tsx to decide whether
 * to auto-switch to the "generate" tab on first load.
 */
function shouldDefaultToGenerate(
  totalQuizzes: number,
  initialAssessmentIntent: unknown
): boolean {
  return totalQuizzes === 0 && !initialAssessmentIntent
}

/**
 * Mirrors the condition used in GenerateTab.tsx to decide whether
 * to show the "no media content" info alert.
 */
function shouldShowNoMediaAlert(
  isLoadingList: boolean,
  loadedMediaItemsLength: number
): boolean {
  return !isLoadingList && loadedMediaItemsLength === 0
}

/**
 * Mirrors the condition used in ResultsTab.tsx to decide whether
 * to show the empty state with a "Take a Quiz" CTA.
 */
function shouldShowEmptyResultsCTA(
  attemptsLength: number,
  hasActiveFilters: boolean
): boolean {
  return attemptsLength === 0 && !hasActiveFilters
}

/**
 * Mirrors the condition used in QuizWorkspace.tsx to decide whether
 * to show the "Try demo quiz" toggle when connected but empty.
 */
function shouldShowDemoToggle(
  isOnline: boolean,
  quizzesUnsupported: boolean,
  totalQuizzes: number
): boolean {
  return isOnline && !quizzesUnsupported && totalQuizzes === 0
}

/**
 * Mirrors the Generate tab label logic in QuizPlayground.tsx that shows
 * "Start here" hint when no quizzes exist.
 */
function generateTabLabel(totalQuizzes: number): string {
  return totalQuizzes === 0 ? "Generate \u2190 Start here" : "Generate"
}

/**
 * Mirrors the beta tooltip text selection in QuizWorkspace.tsx.
 */
function betaTooltipText(isOnline: boolean): string {
  return isOnline
    ? "Quiz Playground is in beta. Features and scoring may evolve. Your quiz data is saved to your server."
    : "Quiz Playground is in beta. Demo answers are local to this session and will not be saved."
}

/**
 * Mirrors the Create tab orientation banner dismiss logic.
 */
const CREATE_ORIENTATION_DISMISSED_KEY = "tldw_quiz_create_orientation_dismissed"

function isOrientationDismissed(): boolean {
  try {
    return window.localStorage.getItem(CREATE_ORIENTATION_DISMISSED_KEY) === "1"
  } catch {
    return false
  }
}

function dismissOrientation(): void {
  try {
    window.localStorage.setItem(CREATE_ORIENTATION_DISMISSED_KEY, "1")
  } catch {
    // localStorage unavailable
  }
}

describe("Quiz FTUX behaviors", () => {
  describe("Task 11: default tab resolution", () => {
    it("should resolve to 'generate' when totalQuizzes is 0 and no assessment intent", () => {
      expect(shouldDefaultToGenerate(0, null)).toBe(true)
    })

    it("should NOT switch when there are quizzes", () => {
      expect(shouldDefaultToGenerate(5, null)).toBe(false)
    })

    it("should NOT switch when there is an assessment intent", () => {
      expect(shouldDefaultToGenerate(0, { startQuizId: 42 })).toBe(false)
    })

    it("should NOT switch when totalQuizzes is > 0 even with no intent", () => {
      expect(shouldDefaultToGenerate(1, null)).toBe(false)
    })
  })

  describe("Task 12: empty-media alert condition", () => {
    it("should show alert when media list is loaded and empty", () => {
      expect(shouldShowNoMediaAlert(false, 0)).toBe(true)
    })

    it("should NOT show alert while still loading", () => {
      expect(shouldShowNoMediaAlert(true, 0)).toBe(false)
    })

    it("should NOT show alert when media items exist", () => {
      expect(shouldShowNoMediaAlert(false, 3)).toBe(false)
    })
  })

  describe("Task 13: Results tab empty state CTA condition", () => {
    it("should show CTA when no attempts and no active filters", () => {
      expect(shouldShowEmptyResultsCTA(0, false)).toBe(true)
    })

    it("should NOT show CTA when there are attempts", () => {
      expect(shouldShowEmptyResultsCTA(2, false)).toBe(false)
    })

    it("should NOT show CTA when filters are active (even with no results)", () => {
      expect(shouldShowEmptyResultsCTA(0, true)).toBe(false)
    })
  })

  describe("P2-E Item 1: demo toggle when connected but empty", () => {
    it("should show demo toggle when online, supported, and zero quizzes", () => {
      expect(shouldShowDemoToggle(true, false, 0)).toBe(true)
    })

    it("should NOT show demo toggle when quizzes exist", () => {
      expect(shouldShowDemoToggle(true, false, 5)).toBe(false)
    })

    it("should NOT show demo toggle when offline", () => {
      expect(shouldShowDemoToggle(false, false, 0)).toBe(false)
    })

    it("should NOT show demo toggle when quizzes unsupported", () => {
      expect(shouldShowDemoToggle(true, true, 0)).toBe(false)
    })
  })

  describe("P2-E Item 2: Generate tab 'Start here' label", () => {
    it("should show 'Start here' hint when no quizzes", () => {
      expect(generateTabLabel(0)).toBe("Generate \u2190 Start here")
    })

    it("should show plain 'Generate' when quizzes exist", () => {
      expect(generateTabLabel(3)).toBe("Generate")
    })
  })

  describe("P2-E Item 4: Create tab orientation banner", () => {
    beforeEach(() => {
      window.localStorage.clear()
    })

    it("should not be dismissed by default", () => {
      expect(isOrientationDismissed()).toBe(false)
    })

    it("should be dismissed after calling dismissOrientation", () => {
      dismissOrientation()
      expect(isOrientationDismissed()).toBe(true)
    })

    it("should persist across calls", () => {
      dismissOrientation()
      // Simulate re-reading from storage
      expect(window.localStorage.getItem(CREATE_ORIENTATION_DISMISSED_KEY)).toBe("1")
    })

    it("should gracefully handle localStorage errors", () => {
      const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new Error("quota exceeded")
      })
      expect(isOrientationDismissed()).toBe(false)
      spy.mockRestore()
    })
  })

  describe("P2-E Item 5: split beta tooltip by context", () => {
    it("should show server-saved message when online", () => {
      expect(betaTooltipText(true)).toContain("saved to your server")
    })

    it("should show local-only message when offline/demo", () => {
      expect(betaTooltipText(false)).toContain("will not be saved")
    })

    it("should NOT mention 'demo' in connected tooltip", () => {
      expect(betaTooltipText(true)).not.toContain("Demo")
    })

    it("should NOT mention 'server' in demo tooltip", () => {
      expect(betaTooltipText(false)).not.toContain("server")
    })
  })
})
