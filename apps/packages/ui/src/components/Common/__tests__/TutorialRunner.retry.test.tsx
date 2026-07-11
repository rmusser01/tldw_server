import { render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { TutorialRunner } from "../TutorialRunner"

let latestJoyrideProps: any = null

const tutorialState = {
  activeTutorialId: "test-tutorial",
  activeStepIndex: 0,
  endTutorial: vi.fn(),
  setStepIndex: vi.fn(),
  markComplete: vi.fn()
}

const tutorialRegistry: Record<string, any> = {
  "test-tutorial": {
    id: "test-tutorial",
    routePattern: "/chat",
    labelKey: "tutorials:test.label",
    labelFallback: "Test tutorial",
    descriptionKey: "tutorials:test.description",
    descriptionFallback: "Test description",
    steps: [
      {
        target: '[data-testid="missing-a"]',
        titleKey: "tutorials:test.stepA.title",
        titleFallback: "Step A",
        contentKey: "tutorials:test.stepA.content",
        contentFallback: "Step A content"
      },
      {
        target: '[data-testid="missing-b"]',
        titleKey: "tutorials:test.stepB.title",
        titleFallback: "Step B",
        contentKey: "tutorials:test.stepB.content",
        contentFallback: "Step B content"
      }
    ]
  }
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("react-joyride", () => {
  const JoyrideMock = (props: any) => {
    latestJoyrideProps = props
    return null
  }

  return {
    default: JoyrideMock,
    STATUS: {
      FINISHED: "finished",
      SKIPPED: "skipped"
    },
    EVENTS: {
      STEP_AFTER: "step:after",
      TARGET_NOT_FOUND: "target:not_found"
    },
    ACTIONS: {
      NEXT: "next",
      PREV: "prev"
    }
  }
})

vi.mock("@/store/tutorials", () => ({
  useActiveTutorial: () => tutorialState
}))

vi.mock("@/tutorials", () => ({
  getTutorialById: (tutorialId: string) => tutorialRegistry[tutorialId]
}))

describe("TutorialRunner retry behavior", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    latestJoyrideProps = null
    tutorialState.activeTutorialId = "test-tutorial"
    tutorialState.activeStepIndex = 0
    tutorialRegistry["test-tutorial"].steps[0].target = '[data-testid="missing-a"]'
    tutorialRegistry["test-tutorial"].steps[1].target = '[data-testid="missing-b"]'
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("retries missing targets before skipping to next step", () => {
    render(<TutorialRunner />)
    const callback = latestJoyrideProps.callback as (data: any) => void

    for (let attempt = 0; attempt < 4; attempt += 1) {
      callback({
        status: "running",
        index: 0,
        type: "target:not_found",
        action: "next"
      })
      vi.advanceTimersByTime(350)
    }

    // Retries keep re-attempting the current step index.
    expect(tutorialState.setStepIndex).toHaveBeenCalledWith(0)

    // After retry budget is exhausted, the runner skips to the next step.
    callback({
      status: "running",
      index: 0,
      type: "target:not_found",
      action: "next"
    })
    expect(tutorialState.setStepIndex).toHaveBeenCalledWith(1)
  })

  it("ends tutorial without completion when last step keeps missing", () => {
    render(<TutorialRunner />)
    const callback = latestJoyrideProps.callback as (data: any) => void

    for (let attempt = 0; attempt < 4; attempt += 1) {
      callback({
        status: "running",
        index: 1,
        type: "target:not_found",
        action: "next"
      })
      vi.advanceTimersByTime(350)
    }

    callback({
      status: "running",
      index: 1,
      type: "target:not_found",
      action: "next"
    })

    expect(tutorialState.markComplete).not.toHaveBeenCalled()
    expect(tutorialState.endTutorial).toHaveBeenCalledTimes(1)
  })

  it("starts on mounted targets and advances without the missing-target fallback", () => {
    tutorialRegistry["test-tutorial"].steps[0].target = '[data-testid="ready-a"]'
    tutorialRegistry["test-tutorial"].steps[1].target = '[data-testid="ready-b"]'
    document.body.insertAdjacentHTML(
      "beforeend",
      '<div data-testid="ready-a">A</div><div data-testid="ready-b">B</div>'
    )

    render(<TutorialRunner />)

    expect(latestJoyrideProps.run).toBe(true)
    expect(latestJoyrideProps.stepIndex).toBe(0)
    for (const step of latestJoyrideProps.steps) {
      expect(document.querySelector(step.target)).toBeInstanceOf(HTMLElement)
    }

    latestJoyrideProps.callback({
      status: "running",
      index: 0,
      type: "step:after",
      action: "next"
    })

    expect(tutorialState.setStepIndex).toHaveBeenCalledWith(1)
    expect(tutorialState.endTutorial).not.toHaveBeenCalled()
  })
})
