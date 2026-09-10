// @vitest-environment jsdom

import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import OptionLayout, { useOptionLayoutShellOverrides } from "../Layout"

const storeMessageOptionMock = vi.hoisted(() =>
  vi.fn(() => ({ historyId: null, serverChatId: null }))
)

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
  useChatSidebar: () => [false]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [""]
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => undefined
}))

vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({
  isMac: false,
  useChatShortcuts: () => undefined,
  useSidebarShortcuts: () => undefined,
  useQuickChatShortcuts: () => undefined,
  useModeNavigationShortcuts: () => undefined
}))

vi.mock("@/components/Layouts/Header", () => ({
  Header: () => <div data-testid="header" />
}))

vi.mock("@/components/Layouts/QuickIngestButton", () => ({
  QuickIngestModalHost: () => null
}))

vi.mock("@/components/Common/QuickChatHelper", () => ({
  QuickChatHelperButton: () => null
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

vi.mock("@/components/Common/PageHelpModal", () => ({
  PageHelpModal: () => null
}))

vi.mock("@/components/Common/TutorialRunner", () => ({
  TutorialRunner: () => null
}))

vi.mock("@/components/Common/TutorialPrompt", () => ({
  TutorialPrompt: () => null
}))

vi.mock("@/components/Common/CommandPaletteHost", () => ({
  CommandPaletteHost: () => null
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

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: unknown) => unknown) =>
    selector({
      historyId: null,
      serverChatId: null
    })
}))

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
  useDemoMode: () => ({ demoMode: false })
}))

describe("OptionLayout shell overrides", () => {
  afterEach(() => {
    delete (
      globalThis as typeof globalThis & {
        __tldwOptionShell?: unknown
      }
    ).__tldwOptionShell
    storeMessageOptionMock.mockClear()
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
