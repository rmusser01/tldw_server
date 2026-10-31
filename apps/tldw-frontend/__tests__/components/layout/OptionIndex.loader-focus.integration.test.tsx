// @vitest-environment jsdom

import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { CommandItem } from "@/components/Common/CommandPalette"
import { useTutorialStore } from "@/store/tutorials"
import WebLayout from "../../../components/layout/WebLayout"
import OptionIndex from "@/routes/option-index"
import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { ConnectionPhase } from "@/types/connection"
import type { FirstRunState } from "@/types/setup-onboarding"

const routeState = vi.hoisted(() => ({
  phase: "unconfigured" as ConnectionPhase,
  firstRunState: null as FirstRunState | null,
  loading: true,
  checkOnce: vi.fn(async () => undefined),
  homeRequested: vi.fn(),
  resolveHome: null as (() => void) | null
}))

const promptState = vi.hoisted(() => ({ commands: [] as CommandItem[] }))
const messageState = vi.hoisted(() => ({
  clearChat: vi.fn(), useOCR: false, chatMode: "normal", setChatMode: vi.fn(),
  webSearch: false, setWebSearch: vi.fn()
}))

// Keep both production layouts, nested override effects, palette hosts/renderers,
// shortcut listeners, and help host/store real. Other shell surfaces are unrelated.
vi.mock("antd", () => ({
  Drawer: () => null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string | { defaultValue?: string }) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue || key })
}))
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback]
}))
vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({ shortcuts: {} })
}))
vi.mock("@/components/Option/Prompt/usePromptPaletteCommands", () => ({
  usePromptPaletteCommands: (_query = "", enabled = true) => enabled ? promptState.commands : []
}))
vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({
  isMac: false,
  useChatShortcuts: () => undefined,
  useSidebarShortcuts: () => undefined,
  useQuickChatShortcuts: () => undefined,
  useModeNavigationShortcuts: () => undefined
}))
vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => messageState }))
vi.mock("@/store/option", () => ({
  useStoreMessageOption: (select: (state: { historyId: null; serverChatId: null }) => unknown) =>
    select({ historyId: null, serverChatId: null })
}))
vi.mock("@/store/quick-chat", () => ({
  useQuickChatStore: () => ({ isOpen: false, setIsOpen: vi.fn() })
}))
vi.mock("@/hooks/useMigration", () => ({ useMigration: () => ({ isLoading: false }) }))
vi.mock("@/hooks/useStorageMigrations", () => ({ useStorageMigrations: () => undefined }))
vi.mock("@/hooks/useLayoutEffectsOwner", () => ({ useLayoutEffectsOwner: () => false }))
vi.mock("@/hooks/useFeatureFlags", () => ({ useChatSidebar: () => [false] }))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }))
vi.mock("@/hooks/useSetting", () => ({ useSetting: () => [""] }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => undefined }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ checkOnce: routeState.checkOnce }),
  useConnectionState: () => ({
    phase: routeState.phase,
    serverUrl: "http://localhost:18811",
    isConnected: routeState.phase === "connected",
    isChecking: false
  }),
  useConnectionUxState: () => ({ isChecking: false })
}))
vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: routeState.firstRunState,
    metadata: null,
    loading: routeState.loading,
    adoptState: vi.fn()
  })
}))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({ useHomeMilestoneScope: () => null }))
vi.mock("@/hooks/usePostOnboardingMediaReadiness", () => ({
  usePostOnboardingMediaReadiness: () => ({ status: "idle" })
}))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
// Defer only the downstream lazy home module. OptionIndex chooses and renders
// its actual setup/home branches, nested layout, banner, and loader fallback.
vi.mock("@/components/Option/CompanionHome", async () => {
  routeState.homeRequested()
  await new Promise<void>((resolve) => { routeState.resolveHome = resolve })
  return { CompanionHomeShell: () => <section data-testid="home-ready">Home ready</section> }
})

vi.mock("@/components/Layouts/Header", () => ({
  Header: () => <header data-testid="shell-header">Shell</header>
}))
vi.mock("@/components/Option/Sidebar", () => ({ Sidebar: () => null }))
vi.mock("@/components/Common/ChatSidebar", () => ({ ChatSidebar: () => null }))
vi.mock("@/components/Layouts/QuickIngestButton", () => ({ QuickIngestModalHost: () => null }))
vi.mock("@/components/Common/QuickChatHelper", () => ({ QuickChatHelperButton: () => null }))
vi.mock("@/components/Common/NotesDock", () => ({ NotesDockHost: () => null }))
vi.mock("@/components/Common/PersonaBuddy", () => ({
  BuddyShellHost: () => null,
  BuddyShellRenderContextProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))
vi.mock("@/components/Common/Settings/CurrentChatModelSettings", () => ({ CurrentChatModelSettings: () => null }))
vi.mock("@/components/Common/Workflow", () => ({ WorkflowIntegrationHost: () => null }))
vi.mock("@/components/Common/confirm-danger", () => ({ useConfirmDanger: () => vi.fn() }))
vi.mock("@/components/Common/BackendRecoveryUiContext", () => ({
  useBackendRecoveryUi: () => ({ fatalBackendRecoveryActive: false })
}))
vi.mock("@web/components/layout/BackendUnavailableModalGate", () => ({ BackendUnavailableModalGate: () => null }))
vi.mock("@web/components/notifications/NotificationLifecycleProvider", () => ({
  NotificationLifecycleProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useNotificationLifecycle: () => ({ state: "idle", unreadCount: 0, tryAgain: vi.fn() })
}))
vi.mock("@web/components/notifications/NotificationToastBridge", () => ({ NotificationToastBridge: () => null }))
vi.mock("@/components/Timeline", () => ({ TimelineModal: () => null }))
vi.mock("@/components/Common/TutorialRunner", () => ({ TutorialRunner: () => null }))
vi.mock("@/components/Common/TutorialPrompt", () => ({ TutorialPrompt: () => null }))
vi.mock("@/components/Common/KeyboardShortcutsModal", () => ({ KeyboardShortcutsModal: () => null }))
vi.mock("@/components/Common/PageHelpModal", () => ({
  PageHelpModal: () => <div role="dialog" aria-label="Page help">Help</div>
}))

beforeEach(() => {
  delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell
  promptState.commands = []
  useTutorialStore.getState().resetProgress()
  HTMLElement.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell
  vi.restoreAllMocks()
})


afterEach(async () => {
  // A red assertion must not leave the downstream lazy import pending.
  await act(async () => { routeState.resolveHome?.() })
})

it("keeps the palette input and focus while actual setup becomes the pending home dashboard", async () => {
  const tree = (mounted: boolean) => <MemoryRouter><WebLayout>
    {mounted ? <OptionIndex /> : <div>Loading route</div>}
  </WebLayout></MemoryRouter>
  // Match the native shell registering before the home route is loaded.
  const view = render(tree(false))
  view.rerender(tree(true))
  const setupLoader = await screen.findByRole("dialog", { name: "Loading setup..." })
  expect(setupLoader).toHaveFocus()
  expect(screen.queryByTestId("shell-header")).not.toBeInTheDocument()
  expect(routeState.homeRequested).not.toHaveBeenCalled()
  fireEvent.keyDown(setupLoader, { key: "k", ctrlKey: true })
  const input = await screen.findByRole("textbox", { name: "Search commands" })
  await waitFor(() => expect(input).toHaveFocus())
  fireEvent.change(input, { target: { value: "go to settings" } })

  routeState.phase = ConnectionPhase.CONNECTED
  routeState.firstRunState = {
    status: "not_started",
    completed_steps: [], skipped_steps: [], step_data: {},
    acknowledged_steps: [], first_chat: { completed: false }
  }
  routeState.loading = false
  view.rerender(tree(true))
  const homeLoader = await screen.findByRole("dialog", { name: "Loading home..." })
  expect(screen.getByTestId("resume-setup-banner")).toBeInTheDocument()
  await waitFor(() => expect(screen.getByTestId("shell-header")).toBeInTheDocument())
  expect(screen.getByText("Preparing your dashboard")).toBeInTheDocument()
  expect(homeLoader).toHaveAttribute("aria-modal", "true")
  expect(homeLoader).toHaveAttribute("aria-busy", "true")
  expect(homeLoader).toHaveClass("fixed", "inset-0")
  await waitFor(() => expect(routeState.homeRequested).toHaveBeenCalledTimes(1))
  expect(setupLoader.isConnected).toBe(false)
  expect(input.isConnected).toBe(true)
  expect(screen.getByRole("textbox", { name: "Search commands" })).toBe(input)
  expect(input).toHaveValue("go to settings")
  expect(input).toHaveFocus()

  await act(async () => { routeState.resolveHome?.() })
  expect(await screen.findByTestId("home-ready")).toBeInTheDocument()
  expect(homeLoader.isConnected).toBe(false)
  expect(screen.getByRole("textbox", { name: "Search commands" })).toBe(input)
  expect(input).toHaveValue("go to settings")
  expect(input).toHaveFocus()
})

describe("existing generic loader focus contract", () => {
  it("takes initial focus by default and exposes its busy accessible status", () => {
    render(<PageAssistLoader label="Initial loading" />)
    const loader = screen.getByRole("dialog", { name: "Initial loading" })
    expect(loader).toHaveFocus()
    expect(loader).toHaveAttribute("aria-modal", "true")
    expect(loader).toHaveAttribute("aria-busy", "true")
    expect(screen.getByRole("progressbar")).toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite")
  })

  it("keeps current focus when autoFocus is false", () => {
    const tree = (loading: boolean) => <><button type="button">Previous focus</button>
      {loading ? <PageAssistLoader label="Non-focusing loading" autoFocus={false} /> : null}</>
    const view = render(tree(false))
    const button = screen.getByRole("button", { name: "Previous focus" })
    button.focus()
    view.rerender(tree(true))
    expect(screen.getByRole("dialog", { name: "Non-focusing loading" })).toBeInTheDocument()
    expect(button).toHaveFocus()
    view.rerender(tree(false))
    expect(button).toHaveFocus()
  })

  it("restores its captured focus target when the default loader unmounts", () => {
    const tree = (loading: boolean) => <><button type="button">Previous focus</button>
      {loading ? <PageAssistLoader label="Restoring loading" /> : null}</>
    const view = render(tree(false))
    const button = screen.getByRole("button", { name: "Previous focus" })
    button.focus()
    view.rerender(tree(true))
    expect(screen.getByRole("dialog", { name: "Restoring loading" })).toHaveFocus()
    view.rerender(tree(false))
    expect(button.isConnected).toBe(true)
    expect(button).toHaveFocus()
  })
})
