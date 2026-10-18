// @vitest-environment jsdom

import React from "react"
import { createPortal } from "react-dom"
import { useQuickChatStore } from "@/store/quick-chat"
import { QuickChatHelperButton } from "@/components/Common/QuickChatHelper"
import { PlaygroundEmpty } from "@/components/Option/Playground/PlaygroundEmpty"
import { useTutorialStore } from "@/store/tutorials"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase } from "@/types/connection"
import { MemoryRouter, useNavigate } from "react-router-dom"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { HistorySelectionProvider, useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import OptionLayout, { useOptionLayoutShellOverrides } from "../Layout"

const storeMessageOptionMock = vi.hoisted(() =>
  vi.fn(() => ({ historyId: null, serverChatId: null }))
)
const sidebarState = vi.hoisted(() => ({ enabled: false, mobile: false }))
const capabilityProbe = vi.hoisted(() => ({
  actualDrawer: false,
  paletteProps: null as {
    onToggleSidebar?: () => void
    onSwitchModel?: () => void
  } | null
}))

vi.mock("antd", async (original) => {
  const actual = await original<typeof import("antd")>()
  return {
    ...actual,
    Drawer: (props: React.ComponentProps<typeof actual.Drawer>) =>
      capabilityProbe.actualDrawer ? (
        <actual.Drawer {...props} />
      ) : props.open ? (
        <div role="dialog">{props.children}</div>
      ) : null
  }
})

vi.mock("@/components/Common/ChatSidebar", () => ({
  ChatSidebar: ({
    onConversationSelected
  }: {
    onConversationSelected?: () => void
  }) => (
    <aside data-testid="chat-sidebar">
      <button onClick={onConversationSelected}>
        Select saved conversation
      </button>
      <QuickChatHelperButton />
    </aside>
  )
}))

vi.mock("@/hooks/useLayoutEffectsOwner", () => ({
  useLayoutEffectsOwner: () => false
}))

vi.mock("@/hooks/useStorageMigrations", () => ({
  useStorageMigrations: () => undefined
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    clearChat: vi.fn(),
    useOCR: false,
    chatMode: "normal",
    setChatMode: vi.fn(),
    webSearch: false,
    setWebSearch: vi.fn()
  })
}))

vi.mock("@/components/Option/Sidebar", () => ({
  Sidebar: () => <aside>Legacy conversations</aside>
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }): string =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue || key
  })
}))

vi.mock("@/hooks/useMigration", () => ({
  useMigration: () => ({ isLoading: false })
}))

vi.mock("@/hooks/useFeatureFlags", () => ({
  useChatSidebar: () => [sidebarState.enabled]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => sidebarState.mobile
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [""]
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => undefined
}))

vi.mock("@/hooks/keyboard/useKeyboardShortcuts", async (original) => {
  const actual =
    await original<typeof import("@/hooks/keyboard/useKeyboardShortcuts")>()
  return {
    isMac: false,
    useChatShortcuts: () => undefined,
    useSidebarShortcuts: actual.useSidebarShortcuts,
    useQuickChatShortcuts: actual.useQuickChatShortcuts,
    useModeNavigationShortcuts: () => undefined
  }
})

vi.mock("@/components/Layouts/Header", () => ({
  Header: () => { const selection = useHistorySelectionContext(); return <div data-testid="header" data-history-controller={selection ? "present" : "absent"} /> }
}))

vi.mock("@/components/Layouts/QuickIngestButton", () => ({
  QuickIngestModalHost: () => null
}))

vi.mock("@/components/Common/QuickChatHelper", () => ({
  QuickChatHelperButton: () => {
    const { isOpen } = useQuickChatStore()
    return isOpen
      ? createPortal(
          <div role="dialog" aria-label="Shared Quick Chat probe">
            Shared helper mounted
          </div>,
          document.body
        )
      : null
  }
}))

vi.mock("@/components/Common/NotesDock", () => ({
  NotesDockHost: () => null
}))

vi.mock("@/components/Common/EventHosts", () => ({
  EventOnlyHosts: () => null
}))

vi.mock("@/components/Timeline", () => ({
  TimelineModal: () => null
}))





vi.mock("@/components/Common/TutorialPrompt", () => ({
  TutorialPrompt: () => null
}))

vi.mock("@/components/Common/CommandPaletteHost", () => ({
  CommandPaletteHost: ({
    commandPaletteProps
  }: {
    commandPaletteProps: NonNullable<typeof capabilityProbe.paletteProps>
  }) => {
    capabilityProbe.paletteProps = commandPaletteProps
    return null
  }
}))

vi.mock("@/components/Option/Prompt/usePromptPaletteCommands", () => ({
  usePromptPaletteCommands: () => []
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(async () => false)
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: () => ""
}))

vi.mock("@/store/option", async (original) => {
  const actual = await original<typeof import("@/store/option")>()
  return {
    ...actual,
    useStoreMessageOption: Object.assign(
      (selector: (state: unknown) => unknown) =>
        selector({ historyId: null, serverChatId: null }),
      {
        getState: actual.useStoreMessageOption.getState,
        setState: actual.useStoreMessageOption.setState,
        subscribe: actual.useStoreMessageOption.subscribe
      }
    )
  }
})

vi.mock("@/utils/settings-return", () => ({
  setSettingsReturnTo: () => undefined
}))

vi.mock("@/utils/ocr", () => ({
  processImageForOCR: () => Promise.resolve("")
}))

vi.mock("@/context/demo-mode", () => ({
  DemoModeProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  useDemoMode: () => ({ demoEnabled: false })
}))

beforeEach(() => {
  const previous = useConnectionStore.getState().state
  useConnectionStore.setState({ state: { ...previous, phase: ConnectionPhase.CONNECTED, isConnected: true } })
  return () => useConnectionStore.setState({ state: previous })
})

describe("OptionLayout shell overrides", () => {
  beforeEach(() => {
    sidebarState.enabled = false
    sidebarState.mobile = false
  })
  afterEach(() => {
    delete (
      globalThis as typeof globalThis & {
        __tldwOptionShell?: unknown
      }
    ).__tldwOptionShell
    storeMessageOptionMock.mockClear()
  })

  it.each([false, true])(
    "fulfills an explicit Find request once the shared shell is revealed (rail=%s)",
    (rail) => {
      sidebarState.enabled = rail
      sidebarState.mobile = true
      const content = (hidden: boolean) => (
        <MemoryRouter initialEntries={["/chat"]}>
          <OptionLayout hideHeader={hidden} hideSidebar={hidden}>
            <div>Draft</div>
          </OptionLayout>
        </MemoryRouter>
      )
      const view = render(content(true))
      act(() => {
        window.dispatchEvent(new Event("tldw:open-chat-sidebar"))
        window.dispatchEvent(new Event("tldw:open-chat-sidebar"))
      })
      expect(screen.queryByRole("dialog")).toBeNull()
      view.rerender(content(false))
      expect(screen.getByRole("dialog")).toBeInTheDocument()
    }
  )

  it.each([false, true])(
    "abandons a hidden shared Find request on route departure (rail=%s)",
    (rail) => {
      sidebarState.enabled = rail
      sidebarState.mobile = true
      function Leave() {
        const navigate = useNavigate()
        return (
          <>
            <button onClick={() => navigate("/notes")}>Leave</button>
            <button onClick={() => navigate("/chat")}>Return</button>
          </>
        )
      }
      const content = (hidden: boolean) => (
        <MemoryRouter initialEntries={["/chat"]}>
          <OptionLayout hideHeader={hidden} hideSidebar={hidden}>
            <Leave />
          </OptionLayout>
        </MemoryRouter>
      )
      const view = render(content(true))
      act(() => window.dispatchEvent(new Event("tldw:open-chat-sidebar")))
      fireEvent.click(screen.getByRole("button", { name: "Leave" }))
      fireEvent.click(screen.getByRole("button", { name: "Return" }))
      view.rerender(content(false))
      expect(screen.queryByRole("dialog")).toBeNull()
    }
  )

  it("abandons a hidden shared Find request when the same path is replaced", () => {
    sidebarState.enabled = true
    sidebarState.mobile = true
    function Replace() {
      const navigate = useNavigate()
      return (
        <button onClick={() => navigate("/chat", { replace: true })}>
          Replace
        </button>
      )
    }
    const content = (hidden: boolean) => (
      <MemoryRouter initialEntries={["/chat"]}>
        <OptionLayout hideHeader={hidden} hideSidebar={hidden}>
          <Replace />
        </OptionLayout>
      </MemoryRouter>
    )
    const view = render(content(true))
    act(() => window.dispatchEvent(new Event("tldw:open-chat-sidebar")))
    fireEvent.click(screen.getByRole("button", { name: "Replace" }))
    view.rerender(content(false))
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("fulfills a pending Find through the current shared mobile host after viewport change", () => {
    sidebarState.enabled = true
    sidebarState.mobile = false
    const content = (hidden: boolean) => (
      <MemoryRouter initialEntries={["/chat"]}>
        <OptionLayout hideHeader={hidden} hideSidebar={hidden}>
          <div>Draft</div>
        </OptionLayout>
      </MemoryRouter>
    )
    const view = render(content(true))
    act(() => window.dispatchEvent(new Event("tldw:open-chat-sidebar")))
    sidebarState.mobile = true
    view.rerender(content(false))
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("closes the mobile drawer on accepted selection without a route change", () => {
    sidebarState.enabled = true
    sidebarState.mobile = true
    render(<MemoryRouter initialEntries={["/chat"]}><OptionLayout><div>Chat</div></OptionLayout></MemoryRouter>)
    act(() => window.dispatchEvent(new CustomEvent("tldw:open-chat-sidebar")))
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Select saved conversation" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("keeps the desktop sidebar mounted after selection", () => {
    sidebarState.enabled = true
    render(<MemoryRouter initialEntries={["/chat"]}><OptionLayout><div>Chat</div></OptionLayout></MemoryRouter>)
    fireEvent.click(screen.getByRole("button", { name: "Select saved conversation" }))
    expect(screen.getByTestId("chat-sidebar")).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("does not clear another shell override if this render never applied one", () => {
    vi.useFakeTimers()
    const externalShell: {
      mounted: boolean
      ownerId: string
      setOverrides?: (overrides: unknown) => void
    } = {
      mounted: true,
      ownerId: "root-shell"
    }
    ;(
      globalThis as typeof globalThis & {
        __tldwOptionShell?: typeof externalShell & {
          setOverrides?: (overrides: unknown) => void
        }
      }
    ).__tldwOptionShell = externalShell

    const { unmount } = render(
      <MemoryRouter>
        <OptionLayout hideHeader>
          <div>Nested content</div>
        </OptionLayout>
      </MemoryRouter>
    )

    const otherOwnerSetOverrides = vi.fn()
    externalShell.setOverrides = otherOwnerSetOverrides

    unmount()

    expect(otherOwnerSetOverrides).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  it("lets route content request header and sidebar shell hiding", async () => {
    const setOverrides = vi.fn()
    const externalShell: {
      mounted: boolean
      ownerId: string
      setOverrides?: (overrides: unknown) => void
    } = {
      mounted: true,
      ownerId: "root-shell",
      setOverrides
    }
    ;(
      globalThis as typeof globalThis & {
        __tldwOptionShell?: typeof externalShell
      }
    ).__tldwOptionShell = externalShell

    function FocusRouteContent() {
      useOptionLayoutShellOverrides({
        hideHeader: true,
        hideSidebar: true
      })

      return <div>Focus route</div>
    }

    const { unmount } = render(
      <MemoryRouter initialEntries={["/chat"]}>
        <FocusRouteContent />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(setOverrides).toHaveBeenCalledWith({
        hideHeader: true,
        hideSidebar: true,
        sourcePath: "/chat"
      })
    })

    unmount()

    expect(setOverrides).toHaveBeenLastCalledWith(null)
  })

  it("clears the same shell setter that accepted route content overrides", async () => {
    const originalSetOverrides = vi.fn()
    const replacementSetOverrides = vi.fn()
    const externalShell: {
      mounted: boolean
      ownerId: string
      setOverrides?: (overrides: unknown) => void
    } = {
      mounted: true,
      ownerId: "root-shell",
      setOverrides: originalSetOverrides
    }
    ;(
      globalThis as typeof globalThis & {
        __tldwOptionShell?: typeof externalShell
      }
    ).__tldwOptionShell = externalShell

    function FocusRouteContent() {
      useOptionLayoutShellOverrides({
        hideHeader: true,
        hideSidebar: true
      })

      return <div>Focus route</div>
    }

    const { unmount } = render(
      <MemoryRouter initialEntries={["/chat"]}>
        <FocusRouteContent />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(originalSetOverrides).toHaveBeenCalledWith({
        hideHeader: true,
        hideSidebar: true,
        sourcePath: "/chat"
      })
    })

    externalShell.setOverrides = replacementSetOverrides

    unmount()

    expect(originalSetOverrides).toHaveBeenLastCalledWith(null)
    expect(replacementSetOverrides).not.toHaveBeenCalled()
  })

  it.each(["/chat", "/settings"])(
    "keeps route content mounted while it requests and releases shell hiding on %s",
    async (pathname) => {
      const mounted = vi.fn()
      const unmounted = vi.fn()

      function RouteContent({ hideShell }: { hideShell: boolean }) {
        useOptionLayoutShellOverrides(
          hideShell ? { hideHeader: true, hideSidebar: true } : null
        )

        React.useEffect(() => {
          mounted()
          return unmounted
        }, [])

        return <div data-testid="route-content">Route content</div>
      }

      const tree = (hideShell: boolean) => (
        <MemoryRouter initialEntries={[pathname]}>
          <OptionLayout>
            <RouteContent hideShell={hideShell} />
          </OptionLayout>
        </MemoryRouter>
      )
      const view = render(tree(false))

      await waitFor(() => expect(mounted).toHaveBeenCalledTimes(1))

      view.rerender(tree(true))

      await waitFor(() => {
        expect(
          view.getByTestId("route-content").parentElement?.className
        ).toContain("items-center")
      })
      expect(mounted).toHaveBeenCalledTimes(1)
      expect(unmounted).not.toHaveBeenCalled()

      view.rerender(tree(false))

      await waitFor(() => {
        expect(
          view.getByTestId("route-content").parentElement?.className
        ).not.toContain("items-center")
      })
      expect(mounted).toHaveBeenCalledTimes(1)
      expect(unmounted).not.toHaveBeenCalled()
    }
  )
})

describe("OptionLayout bypass block (#2889)", () => {
  it("renders a skip link as the first focusable element, targeting the main region", () => {
    const view = render(
      <MemoryRouter>
        <OptionLayout>
          <div data-testid="route-content">Content</div>
        </OptionLayout>
      </MemoryRouter>
    )

    const firstFocusable = view.container.querySelector(
      "a[href], button, [tabindex]:not([tabindex='-1'])"
    ) as HTMLElement
    expect(firstFocusable).toBeTruthy()
    expect(firstFocusable.tagName).toBe("A")
    expect(firstFocusable).toHaveTextContent("Skip to main content")
    expect(firstFocusable).toHaveAttribute("href", "#main-content")

    const main = view.container.querySelector("main")
    expect(main).toHaveAttribute("id", "main-content")
    expect(main).toHaveAttribute("tabindex", "-1")
  })
})

it.each(["/chat", "/settings"])("places the history validator above sidebar consumers on %s", pathname => {
  delete (globalThis as any).__tldwOptionShell
  function Probe() {
    const selection = useHistorySelectionContext()
    return <output data-testid="selection-probe">{selection ? "present" : "absent"}</output>
  }
  const view = render(<MemoryRouter initialEntries={[pathname]}><OptionLayout><Probe /></OptionLayout></MemoryRouter>)
  expect(view.getByTestId("header")).toHaveAttribute("data-history-controller", "present")
  expect(view.getByTestId("selection-probe")).toHaveTextContent("present")
})

it("reuses the shell controller and gives chat route reentry a fresh lifetime", () => {
  delete (globalThis as any).__tldwOptionShell
  const seen: any[] = []
  function Probe() {
    const selection = useHistorySelectionContext()
    const navigate = useNavigate()
    seen.push(selection?.getReference ?? null)
    return <><button onClick={() => navigate("/settings")}>Leave</button><button onClick={() => navigate("/chat")}>Return</button></>
  }
  function Content() {
    const current = useHistorySelectionContext()
    return current ? <HistorySelectionProvider><Probe /></HistorySelectionProvider> : <Probe />
  }
  const view = render(<MemoryRouter initialEntries={["/chat"]}><OptionLayout><Content /></OptionLayout></MemoryRouter>)
  const first = seen.at(-1)
  expect(first).toBeTypeOf("function")
  fireEvent.click(view.getByText("Leave"))
  expect(seen.at(-1)).toBeTypeOf("function")
  expect(seen.at(-1)).not.toBe(first)
  fireEvent.click(view.getByText("Return"))
  expect(seen.at(-1)).toBeTypeOf("function")
  expect(seen.at(-1)).not.toBe(first)
})


const capturePublication = vi.hoisted(() => ({
  callback: null as
    | null
    | ((
        capture: import("@/types/history-selection").HistorySelectionCaptureV1
      ) => void)
}))
vi.mock("@/hooks/chat/useHistorySelection", async (original) => {
  const actual =
    await original<typeof import("@/hooks/chat/useHistorySelection")>()
  return {
    ...actual,
    HistorySelectionProvider: (
      props: Parameters<typeof actual.HistorySelectionProvider>[0]
    ) => {
      if (props.onCapture) capturePublication.callback = props.onCapture
      return React.createElement(actual.HistorySelectionProvider, props)
    }
  }
})

it("the actual outer shell capture cannot republish old messages after synchronous New", async () => {
  const { useStoreMessageOption } =
    await vi.importActual<typeof import("@/store/option")>("@/store/option")
  useStoreMessageOption.setState({
    history: [],
    messages: [],
    serverChatId: "source-chat"
  })
  capturePublication.callback = null
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <OptionLayout>
        <div>Active chat</div>
      </OptionLayout>
    </MemoryRouter>
  )
  expect(capturePublication.callback).not.toBeNull()
  const row = {
    id: "source-answer",
    revision: "v1",
    parent_id: null,
    role: "assistant",
    settled: true
  }
  const capture: import("@/types/history-selection").HistorySelectionCaptureV1 =
    {
      status: "captured",
      purpose: "send",
      storage_context_digest: "context",
      snapshot: {
        version: 1,
        owner_key: "native-owner",
        conversation_id: "source-chat",
        fences: { conversation: "1", history: "1", settings: "0" },
        nodes: [row],
        source_digest: "source",
        interpretation_status: { kind: "parent_graph_v1" },
        storage_context_digest: "context"
      },
      view: {
        view_session_id: "source-view",
        owner_key: "native-owner",
        conversation_id: "source-chat",
        interpretation: { kind: "parent_graph_v1" },
        cursor: { kind: "after_message", message_id: row.id },
        selection_revision: 1
      },
      rows: [row],
      selected_content: [
        {
          id: row.id,
          revision: row.revision,
          message: "Captured earlier reply",
          images: []
        }
      ]
    }
  let replaced = false
  const unsubscribe = useStoreMessageOption.subscribe((state) => {
    if (
      !replaced &&
      state.history.some(
        (message) => message.content === "Captured earlier reply"
      )
    ) {
      replaced = true
      state.startNewChatDraft()
      useStoreMessageOption.setState({
        history: [],
        messages: [],
        serverChatId: null
      })
    }
  })
  try {
    act(() => capturePublication.callback!(capture))
    expect(replaced).toBe(true)
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe(
      "fresh"
    )
    expect(useStoreMessageOption.getState().history).toEqual([])
    expect(useStoreMessageOption.getState().messages).toEqual([])
  } finally {
    unsubscribe()
  }
})

// Actual Host → Page Help → Joyride, with jsdom geometry declared. The real
// Skip/Finish callback happens after the tooltip is removed, unlike control doubles.
it.each(["Close", "Skip tour", "Finish"])(
  "real Joyride %s returns to the original Help opener",
  async (action) => {
    useTutorialStore.getState().resetProgress()
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <OptionLayout hideHeader>
          <PlaygroundEmpty />
          <button data-testid="model-selector">Select a model</button>
          <textarea
            data-testid="chat-input"
            aria-label="Exact tour draft"
            defaultValue={"Real tour draft.  \nSecond line."}
          />
        </OptionLayout>
      </MemoryRouter>
    )
    for (const target of [
      screen.getByRole("button", { name: "Select a model" }),
      screen.getByRole("textbox", { name: "Exact tour draft" })
    ]) {
      vi.spyOn(target, "getBoundingClientRect").mockReturnValue(
        new DOMRect(10, 10, 120, 44)
      )
    }
    const opener = screen.getByRole("button", { name: "Take a quick tour" })
    opener.focus()
    fireEvent.click(opener)
    await screen.findByRole("dialog", { name: "Page Help" })
    const start = screen.getByRole("button", { name: "Start", exact: true })
    start.focus()
    fireEvent.click(start)
    await screen.findByRole("alertdialog", { name: "Choose a Model" })
    if (action === "Finish") {
      for (let step = 1; step < 5; step++) {
        const next = await screen.findByRole("button", {
          name: `Next (Step ${step} of 5)`
        })
        next.focus()
        fireEvent.click(next)
      }
    }
    const exit = await screen.findByRole("button", {
      name: action,
      exact: true
    })
    exit.focus()
    fireEvent.click(exit)
    await waitFor(() =>
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument()
    )
    await waitFor(() => expect(opener).toHaveFocus())
    expect(
      screen.getByRole("textbox", { name: "Exact tour draft" })
    ).toHaveValue("Real tour draft.  \nSecond line.")
    expect(useTutorialStore.getState().completedTutorials).toEqual(
      action === "Finish" ? ["playground-basics"] : []
    )
    act(() => useTutorialStore.getState().resetProgress())
  }
)

it("real Joyride cannot adopt BODY after outside focus even on explicit Skip", async () => {
  useTutorialStore.getState().resetProgress()
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <OptionLayout hideHeader>
        <PlaygroundEmpty />
        <button data-testid="model-selector">Select a model</button>
        <textarea data-testid="chat-input" aria-label="Outside tour input" />
      </OptionLayout>
    </MemoryRouter>
  )
  vi.spyOn(
    screen.getByRole("button", { name: "Select a model" }),
    "getBoundingClientRect"
  ).mockReturnValue(new DOMRect(10, 10, 120, 44))
  const opener = screen.getByRole("button", { name: "Take a quick tour" })
  opener.focus()
  fireEvent.click(opener)
  await screen.findByRole("dialog", { name: "Page Help" })
  const start = screen.getByRole("button", { name: "Start", exact: true })
  start.focus()
  fireEvent.click(start)
  await screen.findByRole("alertdialog", { name: "Choose a Model" })
  const skip = screen.getByRole("button", { name: "Skip tour" })
  skip.focus()
  const outside = screen.getByRole("textbox", { name: "Outside tour input" })
  outside.focus()
  outside.blur()
  const focus = vi.spyOn(opener, "focus")
  fireEvent.click(skip)
  await waitFor(() => expect(useTutorialStore.getState().activeRun).toBeNull())
  await act(
    async () => new Promise((resolve) => window.requestAnimationFrame(resolve))
  )
  expect(focus).not.toHaveBeenCalled()
  expect(useTutorialStore.getState().completedTutorials).toEqual([])
})

it("starts the real shared tutorial runner from Tour while the full extension header is hidden", async () => {
  useTutorialStore.getState().resetProgress()
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <OptionLayout hideHeader>
        <PlaygroundEmpty />
        <button data-testid="model-selector">Select a model</button>
        <textarea data-testid="chat-input" aria-label="Test composer" />
      </OptionLayout>
    </MemoryRouter>
  )
  const target = screen.getByRole("button", { name: "Select a model" })
  // jsdom does not calculate layout; the native runner gate measures real targets.
  vi.spyOn(target, "getBoundingClientRect").mockReturnValue(
    new DOMRect(10, 10, 120, 44)
  )
  fireEvent.click(screen.getByRole("button", { name: "Take a quick tour" }))
  const help = await screen.findByRole("dialog", { name: "Page Help" })
  fireEvent.click(screen.getByRole("button", { name: "Start", exact: true }))
  expect(help).not.toBeInTheDocument()
  const tour = await screen.findByRole("alertdialog", {
    name: "Choose a Model"
  })
  expect(tour).toBeVisible()
  const next = screen.getByRole("button", { name: "Next (Step 1 of 5)" })
  next.focus()
  fireEvent.keyDown(next, { key: "?", shiftKey: true })
  expect(
    screen.queryByRole("dialog", { name: "Page Help" })
  ).not.toBeInTheDocument()
  expect(tour).toBeVisible()
  fireEvent.click(
    tour.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!
  )
  await waitFor(() =>
    expect(
      screen.queryByRole("alertdialog", { name: "Choose a Model" })
    ).not.toBeInTheDocument()
  )
  expect(useTutorialStore.getState().completedTutorials).not.toContain(
    "playground-basics"
  )
  act(() => useTutorialStore.getState().resetProgress())
})


it.each([
  { hideHeader: true, hideSidebar: false },
  { hideHeader: false, hideSidebar: true }
])(
  "shared shell disables hidden-host keyboard, palette and reference commands with %j",
  async (overrides) => {
    const { useQuickChatStore } = await import("@/store/quick-chat")
    useQuickChatStore.getState().setIsOpen(false)
    useTutorialStore.getState().closeHelpModal()
    sidebarState.enabled = true
    sidebarState.mobile = true
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <OptionLayout {...overrides}>
          <textarea
            aria-label="Exact shared draft"
            defaultValue="Shared draft.  "
          />
        </OptionLayout>
      </MemoryRouter>
    )
    fireEvent.keyDown(document, { key: "b", ctrlKey: true })
    fireEvent.keyDown(document, { key: "h", ctrlKey: true, shiftKey: true })
    expect(useQuickChatStore.getState().isOpen).toBe(false)
    expect(capabilityProbe.paletteProps?.onToggleSidebar).toBeUndefined()
    if (overrides.hideHeader)
      expect(capabilityProbe.paletteProps?.onSwitchModel).toBeUndefined()
    else
      expect(capabilityProbe.paletteProps?.onSwitchModel).toBeTypeOf("function")
    act(() => window.dispatchEvent(new CustomEvent("tldw:open-chat-sidebar")))
    expect(screen.queryByTestId("chat-sidebar")).not.toBeInTheDocument()
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    const help = await screen.findByRole("dialog", { name: "Page Help" })
    expect(within(help).queryByText("Toggle sidebar")).not.toBeInTheDocument()
    expect(
      within(help).queryByText("Toggle Quick Chat Helper")
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("textbox", { name: "Exact shared draft" })
    ).toHaveValue("Shared draft.  ")
    fireEvent.keyDown(help, { key: "Escape" })
  }
)


it("shared shell demand-mounts its existing helper inside a cold real mobile Drawer", async () => {
  capabilityProbe.actualDrawer = true
  sidebarState.enabled = true
  sidebarState.mobile = true
  useQuickChatStore.getState().setIsOpen(false)
  try {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <OptionLayout>
          <textarea
            aria-label="Shared mobile draft"
            defaultValue="Mobile shared.  "
          />
        </OptionLayout>
      </MemoryRouter>
    )
    expect(screen.queryByTestId("chat-sidebar")).not.toBeInTheDocument()
    fireEvent.keyDown(document, { key: "h", ctrlKey: true, shiftKey: true })
    expect(
      await screen.findByRole("dialog", { name: "Shared Quick Chat probe" })
    ).toBeVisible()
    expect(screen.getByTestId("chat-sidebar")).not.toBeVisible()
    expect(
      screen.getByRole("textbox", { name: "Shared mobile draft" })
    ).toHaveValue("Mobile shared.  ")
    fireEvent.keyDown(document, { key: "h", ctrlKey: true, shiftKey: true })
    expect(
      screen.queryByRole("dialog", { name: "Shared Quick Chat probe" })
    ).not.toBeInTheDocument()
  } finally {
    capabilityProbe.actualDrawer = false
    useQuickChatStore.getState().setIsOpen(false)
  }
})
