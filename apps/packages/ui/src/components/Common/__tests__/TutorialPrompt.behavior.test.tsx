import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TutorialRun } from "@/store/tutorials"
import { TutorialPrompt } from "../TutorialPrompt"
import { useStoreMessageOption } from "@/store/option"
import {
  HistorySelectionContext,
  type HistorySelectionController
} from "@/hooks/chat/useHistorySelection"

const infoMock = vi.fn()
const destroyMock = vi.fn()
const markPromptSeenMock = vi.fn()
const startTutorialMock = vi.fn()

const tutorialStoreState = {
  completedTutorials: [] as string[],
  hasSeenPromptForPage: vi.fn(() => false),
  markPromptSeen: markPromptSeenMock,
  startTutorial: startTutorialMock,
  isHelpModalOpen: false,
  activeTutorialId: null as string | null,
  activeRun: null as TutorialRun | null
}

type TutorialRouteOptions = {
  completedTutorialIds?: readonly string[]
}

const hasTutorialsForRouteMock = vi.fn<
  (route: string, options?: TutorialRouteOptions) => boolean
>(() => true)
const getPrimaryTutorialForRouteMock = vi.fn<
  (
    route: string,
    options?: TutorialRouteOptions
  ) => {
    id: string
    labelKey: string
    labelFallback: string
  }
>(() => ({
  id: "playground-basics",
  labelKey: "tutorials:playground.basics.label",
  labelFallback: "Chat Basics"
}))

let capturedNavigate: ReturnType<typeof useNavigate> | null = null

const NavigatorBridge = () => {
  capturedNavigate = useNavigate()
  return null
}

const translate = (
  _key: string,
  fallback?: string | { defaultValue?: string }
) => (typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key)
const notificationApi = { info: infoMock, destroy: destroyMock }
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: translate }) }))

vi.mock("antd", () => ({
  notification: {
    useNotification: () => [
      notificationApi,
      <div key="tutorial-prompt-holder" data-testid="tutorial-prompt-holder" />
    ]
  },
  Button: ({
    children,
    ...props
  }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
  Space: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialStore: Object.assign(
    (selector: (state: typeof tutorialStoreState) => unknown) =>
      selector(tutorialStoreState),
    { getState: () => tutorialStoreState }
  )
}))

vi.mock("@/tutorials", () => ({
  hasTutorialsForRoute: (route: string, options?: TutorialRouteOptions) =>
    hasTutorialsForRouteMock(route, options),
  getPrimaryTutorialForRoute: (route: string, options?: TutorialRouteOptions) =>
    getPrimaryTutorialForRouteMock(route, options)
}))

describe("TutorialPrompt behavior", () => {
  it.each(["dismiss", "close", "unmount", "same-path navigation"])(
    "an old notification Start refuses %s",
    (boundary) => {
      const mounted = render(
        <MemoryRouter initialEntries={["/chat"]}>
          <NavigatorBridge />
          <TutorialPrompt />
        </MemoryRouter>
      )
      act(() => vi.advanceTimersByTime(2000))
      const notification = infoMock.mock.calls[0][0]
      const originalStart = notification.btn.props.children[1].props
        .onClick as () => void
      if (boundary === "dismiss")
        act(() => notification.btn.props.children[0].props.onClick())
      else if (boundary === "close") act(() => notification.onClose())
      else if (boundary === "unmount") mounted.unmount()
      else act(() => capturedNavigate?.("/chat?find=2#help"))
      act(originalStart)
      expect(startTutorialMock).not.toHaveBeenCalled()
    }
  )

  it.each(["before delay", "after display"])(
    "unchanged source rerender %s retains one prompt",
    (timing) => {
      const content = () => (
        <MemoryRouter initialEntries={["/chat"]}>
          <TutorialPrompt />
        </MemoryRouter>
      )
      const mounted = render(content())
      if (timing === "after display") act(() => vi.advanceTimersByTime(2000))
      mounted.rerender(content())
      act(() => vi.advanceTimersByTime(2000))
      expect(infoMock).toHaveBeenCalledOnce()
      expect(destroyMock).not.toHaveBeenCalled()
    }
  )

  it.each(["New", "H1"])(
    "a prompt's original Start refuses replacement %s",
    (boundary) => {
      let epoch = 1
      const selection = {
        fence: () => {
          const captured = epoch
          return () => captured === epoch
        }
      } as unknown as HistorySelectionController
      render(
        <MemoryRouter initialEntries={["/chat"]}>
          <HistorySelectionContext.Provider value={selection}>
            <TutorialPrompt />
          </HistorySelectionContext.Provider>
        </MemoryRouter>
      )
      act(() => vi.advanceTimersByTime(2000))
      const original = infoMock.mock.calls[0][0].btn
      if (boundary === "New")
        act(() => useStoreMessageOption.getState().startNewChatDraft())
      else epoch++
      render(original)
      fireEvent.click(screen.getByRole("button", { name: "Start tour" }))
      expect(startTutorialMock).not.toHaveBeenCalled()
      expect(destroyMock).not.toHaveBeenCalled()
      expect(markPromptSeenMock).not.toHaveBeenCalled()
    }
  )

  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    tutorialStoreState.hasSeenPromptForPage.mockReturnValue(false)
    tutorialStoreState.completedTutorials = []
    tutorialStoreState.isHelpModalOpen = false
    tutorialStoreState.activeTutorialId = null
    tutorialStoreState.activeRun = null
    startTutorialMock.mockImplementation((_id, focusReturn) => {
      tutorialStoreState.activeRun = { id: 1, focusReturn }
    })
    capturedNavigate = null
  })

  afterEach(() => {
    startTutorialMock.mock.calls.forEach((call) => call[1]?.dispose?.())
    vi.useRealTimers()
  })

  it("shows a prompt with a single global notification key", () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <Routes>
          <Route
            path="*"
            element={
              <>
                <NavigatorBridge />
                <TutorialPrompt />
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    )

    act(() => {
      vi.advanceTimersByTime(2000)
    })

    expect(infoMock).toHaveBeenCalledTimes(1)
    expect(infoMock.mock.calls[0][0]?.key).toBe("tutorial-prompt-global")
  })

  it("passes completed tutorial IDs into tutorial route lookups", () => {
    tutorialStoreState.completedTutorials = ["getting-started"]

    render(
      <MemoryRouter initialEntries={["/knowledge"]}>
        <Routes>
          <Route
            path="*"
            element={
              <>
                <NavigatorBridge />
                <TutorialPrompt />
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    )

    act(() => {
      vi.advanceTimersByTime(2000)
    })

    const expectedOptions = {
      completedTutorialIds: ["getting-started"]
    }
    expect(hasTutorialsForRouteMock).toHaveBeenCalledWith(
      "/knowledge",
      expectedOptions
    )
    expect(getPrimaryTutorialForRouteMock).toHaveBeenCalledWith(
      "/knowledge",
      expectedOptions
    )
  })

  it("avoids rapid duplicate prompts across route changes", () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <Routes>
          <Route
            path="*"
            element={
              <>
                <NavigatorBridge />
                <TutorialPrompt />
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    )

    act(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(infoMock).toHaveBeenCalledTimes(1)

    act(() => {
      capturedNavigate?.("/media")
    })

    expect(destroyMock).toHaveBeenCalledWith("tutorial-prompt-global")

    // Cooldown prevents immediate second prompt spam.
    act(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(infoMock).toHaveBeenCalledTimes(1)

    // After cooldown window, prompt may appear for the new route.
    act(() => {
      vi.advanceTimersByTime(10000)
    })
    expect(infoMock).toHaveBeenCalledTimes(2)
  })

  // XP-05: opening a note writes /notes?note=<id> to the address. A same-page
  // address change before the prompt appears must not cancel it for the rest
  // of the session; it is shown once the page settles.
  it("still shows the prompt once after a same-page address change before it appears", () => {
    render(
      <MemoryRouter initialEntries={["/notes"]}>
        <NavigatorBridge />
        <TutorialPrompt />
      </MemoryRouter>
    )

    act(() => {
      vi.advanceTimersByTime(500)
    })
    act(() => {
      capturedNavigate?.("/notes?note=n1")
    })
    act(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(infoMock).toHaveBeenCalledTimes(1)

    act(() => {
      capturedNavigate?.("/notes?note=n2")
    })
    act(() => {
      vi.advanceTimersByTime(15000)
    })
    expect(infoMock).toHaveBeenCalledTimes(1)
  })
})
