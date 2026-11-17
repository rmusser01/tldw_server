import { act, render as renderUI } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { TutorialRunner } from "../TutorialRunner"

const render = (ui: React.ReactNode) => {
  const mounted = renderUI(<MemoryRouter>{ui}</MemoryRouter>)
  return {
    ...mounted,
    rerender: (next: React.ReactNode) =>
      mounted.rerender(<MemoryRouter>{next}</MemoryRouter>)
  }
}
let latestJoyrideProps: any = null

const tutorialState = {
  activeTutorialId: "test-tutorial",
  activeRun: { id: 1 },
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
      TARGET_NOT_FOUND: "target:not_found",
      TOOLTIP: "tooltip"
    },
    ACTIONS: {
      NEXT: "next",
      PREV: "prev",
      CLOSE: "close"
    }
  }
})

vi.mock("@/store/tutorials", () => ({
  useActiveTutorial: () => tutorialState,
  useTutorialStore: { getState: () => tutorialState }
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
    tutorialState.activeRun = { id: tutorialState.activeRun.id + 1 }
    tutorialRegistry["test-tutorial"].steps[0].target = '[data-testid="missing-a"]'
    tutorialRegistry["test-tutorial"].steps[1].target = '[data-testid="missing-b"]'
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it.each([false, true])(
    "uses the first visible target alternative rather than DOM order (primary hidden=%s)",
    (hidden) => {
      tutorialRegistry["test-tutorial"].steps[0].target =
        '[data-testid="primary-model"], [data-testid="fallback-input"]'
      // jsdom has no layout; both are measurable, but a hidden primary stays hidden.
      vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(
        new DOMRect(10, 10, 120, 44)
      )
      const view = render(
        <>
          <textarea data-testid="fallback-input" />
          <button
            data-testid="primary-model"
            style={{ display: hidden ? "none" : "block" }}>
            Model
          </button>
        </>
      )
      render(<TutorialRunner />)
      const expected = view.getByTestId(
        hidden ? "fallback-input" : "primary-model"
      )
      expect(latestJoyrideProps.steps[0].target).toBe(expected)
    }
  )

  it("re-resolves the original alternatives when a previously mounted target is removed", () => {
    tutorialRegistry["test-tutorial"].steps[0].target =
      '[data-testid="primary-model"], [data-testid="fallback-input"]'
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(
      new DOMRect(10, 10, 120, 44)
    )
    const targets = render(
      <>
        <textarea data-testid="fallback-input" />
        <button data-testid="primary-model">Model</button>
      </>
    )
    const fallback = targets.getByTestId("fallback-input")
    const scroll = vi.fn()
    fallback.scrollIntoView = scroll
    render(<TutorialRunner />)
    targets.rerender(
      <>
        <textarea data-testid="fallback-input" />
      </>
    )
    act(() => {
      latestJoyrideProps.callback({
        status: "running",
        index: 0,
        type: "target:not_found",
        action: "next"
      })
      vi.advanceTimersByTime(350)
    })
    expect(scroll).toHaveBeenCalled()
    expect(latestJoyrideProps.steps[0].target).toBe(fallback)
  })

  it.each(["running", "finished"])(
    "closes an explicit tour Close without granting completion (status=%s)",
    (status) => {
      render(<TutorialRunner />)
      latestJoyrideProps.callback({
        status,
        index: 0,
        type: "step:after",
        action: "close"
      })
      expect(tutorialState.endTutorial).toHaveBeenCalledTimes(1)
      expect(tutorialState.setStepIndex).not.toHaveBeenCalled()
      expect(tutorialState.markComplete).not.toHaveBeenCalled()
    }
  )

  it("records completion after the explicit Finish action", () => {
    render(<TutorialRunner />)
    latestJoyrideProps.callback({
      status: "finished",
      index: 1,
      type: "step:after",
      action: "next"
    })
    expect(tutorialState.markComplete).toHaveBeenCalledWith("test-tutorial", tutorialState.activeRun)
    expect(tutorialState.endTutorial).toHaveBeenCalledTimes(1)
  })

  it("polls a missing target without remounting, then skips to the next step", () => {
    render(<TutorialRunner />)
    const callback = latestJoyrideProps.callback as (data: any) => void

    callback({
      status: "running",
      index: 0,
      type: "target:not_found",
      action: "next"
    })
    // Joyride reports a missing target once; repeated reports must not stack
    // extra retry chains.
    callback({
      status: "running",
      index: 0,
      type: "target:not_found",
      action: "next"
    })

    act(() => vi.advanceTimersByTime(350 * 3))
    // Still polling: no step change yet (NO-N1).
    expect(tutorialState.setStepIndex).not.toHaveBeenCalled()

    // After the retry budget is exhausted, the runner skips to the next step.
    act(() => vi.advanceTimersByTime(350))
    expect(tutorialState.setStepIndex).toHaveBeenCalledTimes(1)
    expect(tutorialState.setStepIndex).toHaveBeenCalledWith(1, tutorialState.activeRun)
  })

  it("records completion when the last target never mounts after the user saw the tour", () => {
    render(<TutorialRunner />)
    const callback = latestJoyrideProps.callback as (data: any) => void
    callback({ status: "running", index: 0, type: "tooltip", action: "update" })
    callback({
      status: "running",
      index: 1,
      type: "target:not_found",
      action: "next"
    })
    act(() => vi.advanceTimersByTime(350 * 4))
    expect(tutorialState.markComplete).toHaveBeenCalledWith(
      "test-tutorial",
      tutorialState.activeRun
    )
    expect(tutorialState.endTutorial).toHaveBeenCalledTimes(1)
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

    expect(tutorialState.setStepIndex).toHaveBeenCalledWith(1, tutorialState.activeRun)
    expect(tutorialState.endTutorial).not.toHaveBeenCalled()
  })

  it.each(["close", "finish", "step"])(
    "ignores a retained %s callback after same-ID restart before rerender",
    (kind) => {
      render(<TutorialRunner />)
      const old = latestJoyrideProps.callback
      tutorialState.activeRun = { id: tutorialState.activeRun.id + 1 }
      old({
        status: kind === "finish" ? "finished" : "running",
        index: 0,
        type: "step:after",
        action: kind === "close" ? "close" : "next"
      })
      expect(tutorialState.endTutorial).not.toHaveBeenCalled()
      expect(tutorialState.markComplete).not.toHaveBeenCalled()
      expect(tutorialState.setStepIndex).not.toHaveBeenCalled()
    }
  )

  it("ignores a queued old target timer when the same tutorial is restarted before rerender", () => {
    render(<TutorialRunner />)
    latestJoyrideProps.callback({
      status: "running",
      index: 0,
      type: "target:not_found",
      action: "next"
    })
    tutorialState.activeRun = { id: tutorialState.activeRun.id + 1 }
    act(() => vi.advanceTimersByTime(350))
    expect(tutorialState.setStepIndex).not.toHaveBeenCalled()
  })

  it("does not end a replacement synchronously started by completion publication", () => {
    render(<TutorialRunner />)
    tutorialState.markComplete.mockImplementationOnce(() => {
      tutorialState.activeRun = { id: tutorialState.activeRun.id + 1 }
    })
    latestJoyrideProps.callback({
      status: "finished",
      index: 1,
      type: "step:after",
      action: "next"
    })
    expect(tutorialState.markComplete).toHaveBeenCalledOnce()
    expect(tutorialState.endTutorial).not.toHaveBeenCalled()
  })
})
