/**
 * TutorialRunner against the real react-joyride (NO-N1, NO-02).
 *
 * TutorialRunner.retry.test.tsx mocks react-joyride, so it cannot see how the
 * real library behaves when a step target is missing. These tests drive the
 * real tour through the DOM.
 */
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TutorialRunner } from "../TutorialRunner"
import { useTutorialStore } from "@/store/tutorials"

const step = (id: string, extra: Record<string, unknown> = {}) => ({
  target: `[data-testid="${id}"]`,
  titleKey: `tutorials:test.${id}.title`,
  titleFallback: `Title ${id}`,
  contentKey: `tutorials:test.${id}.content`,
  contentFallback: `Content ${id}`,
  placement: "bottom" as const,
  disableBeacon: true,
  ...extra
})

const tutorialRegistry: Record<string, { id: string; steps: ReturnType<typeof step>[] } & Record<string, unknown>> = {}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/tutorials", () => ({
  getTutorialById: (tutorialId: string) => tutorialRegistry[tutorialId]
}))

const defineTutorial = (id: string, steps: ReturnType<typeof step>[]) => {
  tutorialRegistry[id] = {
    id,
    routePattern: "/test",
    labelKey: "tutorials:test.label",
    labelFallback: "Test tour",
    descriptionKey: "tutorials:test.description",
    descriptionFallback: "Test tour",
    steps
  }
}

const renderTour = (targets: React.ReactNode) =>
  render(
    <MemoryRouter>
      {targets}
      <TutorialRunner />
    </MemoryRouter>
  )

const startTour = (id: string) => {
  act(() => {
    useTutorialStore.getState().startTutorial(id)
  })
}

const clickPrimary = () => {
  const tooltip = document.querySelector(".react-joyride__tooltip")
  const primary = tooltip?.querySelector<HTMLButtonElement>(
    'button[data-action="primary"]'
  )
  if (!primary) throw new Error("No tour tooltip primary button")
  fireEvent.click(primary)
}

describe("TutorialRunner with the real react-joyride", { timeout: 30_000 }, () => {
  beforeEach(() => {
    localStorage.clear()
    useTutorialStore.setState({
      completedTutorials: [],
      seenPromptPages: [],
      activeTutorialId: null,
      activeRun: null,
      activeStepIndex: 0
    })
    // jsdom has no layout; give every element a measurable box.
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(
      new DOMRect(10, 10, 120, 40)
    )
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
  })

  afterEach(() => {
    act(() => {
      useTutorialStore.getState().endTutorial()
    })
    cleanup()
    vi.restoreAllMocks()
  })

  it("skips a step whose target never mounts and still records completion (NO-N1)", async () => {
    defineTutorial("missing-middle", [step("a"), step("b"), step("c")])
    renderTour(
      <>
        <div data-testid="a">A</div>
        <div data-testid="c">C</div>
      </>
    )
    startTour("missing-middle")

    expect(await screen.findByText("Content a")).toBeInTheDocument()
    clickPrimary()

    // Step b's target is missing: the tour must move on to c rather than
    // hang with no tooltip.
    expect(
      await screen.findByText("Content c", {}, { timeout: 8_000 })
    ).toBeInTheDocument()
    clickPrimary()

    await waitFor(() => {
      expect(useTutorialStore.getState().completedTutorials).toContain(
        "missing-middle"
      )
    })
    expect(useTutorialStore.getState().activeRun).toBeNull()
  })

  it("shows a step whose target mounts while the runner is retrying", async () => {
    // jsdom has no scrollIntoView; the runner scrolls a late target into view.
    const scrollIntoView = vi.fn()
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      configurable: true,
      writable: true,
      value: scrollIntoView
    })
    const lateTarget = document.createElement("div")
    lateTarget.dataset.testid = "late"
    lateTarget.textContent = "Late"
    try {
      defineTutorial("late-target", [step("a"), step("late"), step("c")])
      renderTour(
        <>
          <div data-testid="a">A</div>
          <div data-testid="c">C</div>
        </>
      )
      startTour("late-target")
      expect(await screen.findByText("Content a")).toBeInTheDocument()
      clickPrimary()

      // The target appears shortly after the step was requested, without
      // re-rendering the runner: only its own retry can pick it up.
      await new Promise((resolve) => setTimeout(resolve, 200))
      document.body.appendChild(lateTarget)

      expect(
        await screen.findByText("Content late", {}, { timeout: 8_000 })
      ).toBeInTheDocument()
      expect(scrollIntoView).toHaveBeenCalled()
      expect(screen.queryByText("Content c")).not.toBeInTheDocument()
    } finally {
      lateTarget.remove()
      delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView
    }
  })

  it("counts only the steps it can show when optional targets are absent (NO-02)", async () => {
    defineTutorial("optional-step", [
      step("a"),
      step("needs-selection", { optional: true }),
      step("c")
    ])
    renderTour(
      <>
        <div data-testid="a">A</div>
        <div data-testid="c">C</div>
      </>
    )
    startTour("optional-step")

    expect(await screen.findByText("Content a")).toBeInTheDocument()
    const tooltip = document.querySelector(".react-joyride__tooltip")
    // "Step 1 of 2", not "of 3": the optional step has no target to show.
    expect(tooltip?.textContent).toMatch(/1\s*of\s*2/)
  })

  it("puts the window scroll position back when the tour is skipped (NO-02)", async () => {
    // jsdom does not scroll; model the window scroll position.
    let pageY = 0
    vi.spyOn(window, "scrollY", "get").mockImplementation(() => pageY)
    vi.spyOn(window, "scrollTo").mockImplementation(((
      xOrOptions?: number | ScrollToOptions,
      y?: number
    ) => {
      pageY =
        typeof xOrOptions === "object"
          ? Number(xOrOptions?.top ?? pageY)
          : Number(y ?? pageY)
    }) as typeof window.scrollTo)

    defineTutorial("scroll-restore", [step("a"), step("c")])
    renderTour(
      <>
        <div data-testid="a">A</div>
        <div data-testid="c">C</div>
      </>
    )
    startTour("scroll-restore")
    expect(await screen.findByText("Content a")).toBeInTheDocument()

    // The tour scrolled the page to reach a target.
    pageY = 315
    const skip = document.querySelector<HTMLButtonElement>(
      '.react-joyride__tooltip button[data-action="skip"]'
    )
    if (!skip) throw new Error("No skip button")
    fireEvent.click(skip)

    await waitFor(() => {
      expect(useTutorialStore.getState().activeRun).toBeNull()
    })
    expect(pageY).toBe(0)
  })

  it("leaves pane scrolling alone on viewport-framed routes (NO-02)", async () => {
    defineTutorial("framed-pane", [step("in-pane"), step("c")])
    render(
      <MemoryRouter initialEntries={["/notes"]}>
        <div data-testid="pane" style={{ overflow: "auto" }}>
          <div data-testid="in-pane">In pane</div>
        </div>
        <div data-testid="c">C</div>
        <TutorialRunner />
      </MemoryRouter>
    )
    startTour("framed-pane")
    expect(await screen.findByText("Content in-pane")).toBeInTheDocument()
    // Joyride's scroll-parent fix sets overflow: initial on a pane that is
    // not scrolling yet, which breaks the pane once its content grows.
    expect(screen.getByTestId("pane").style.overflow).toBe("auto")
  })
})
