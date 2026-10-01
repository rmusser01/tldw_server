// @vitest-environment jsdom

import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import SharedLayout from "@/components/Layouts/Layout"
import type { CommandItem } from "@/components/Common/CommandPalette"
import { useTutorialStore } from "@/store/tutorials"
import WebLayout from "../../../components/layout/WebLayout"

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
  useConnectionActions: () => ({ checkOnce: vi.fn(async () => undefined) }),
  useConnectionState: () => ({ phase: "connected", isConnected: true, isChecking: false }),
  useConnectionUxState: () => ({ isChecking: false })
}))
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

const layouts = [
  { name: "WebLayout", Layout: WebLayout },
  { name: "SharedLayout", Layout: SharedLayout }
]

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

for (const { name, Layout } of layouts) {
  describe(`${name} command palette lifetime`, () => {
    const tree = (hideHeader: boolean, nestedRoute = true) => (
      <MemoryRouter initialEntries={["/"]}>
        <Layout>
          {nestedRoute ? (
            <SharedLayout hideHeader={hideHeader} hideSidebar={hideHeader}>
              <button type="button">Focus before palette</button>
            </SharedLayout>
          ) : <div>Loading route</div>}
        </Layout>
      </MemoryRouter>
    )

    it.each([true, false])("preserves the open palette through nested hideHeader=%s changes", async (initiallyHidden) => {
      const addListener = vi.spyOn(window, "addEventListener")
      const removeListener = vi.spyOn(window, "removeEventListener")
      // Native home loads its nested route after the outer shell registers itself.
      const view = render(tree(false, false))
      view.rerender(tree(initiallyHidden))
      await waitFor(() => expect(Boolean(screen.queryByTestId("shell-header"))).toBe(!initiallyHidden))
      const trigger = screen.getByRole("button", { name: "Focus before palette" })
      trigger.focus()
      fireEvent.keyDown(trigger, { key: "k", ctrlKey: true })
      const input = await screen.findByRole("textbox", { name: "Search commands" })
      await waitFor(() => expect(input).toHaveFocus())
      fireEvent.change(input, { target: { value: "go to settings" } })

      view.rerender(tree(!initiallyHidden))
      await waitFor(() => expect(Boolean(screen.queryByTestId("shell-header"))).toBe(initiallyHidden))
      expect(input.isConnected).toBe(true)
      expect(screen.getByRole("textbox", { name: "Search commands" })).toBe(input)
      expect(input).toHaveValue("go to settings")
      expect(input).toHaveFocus()
      expect(screen.getAllByRole("dialog", { name: "Command Palette" })).toHaveLength(1)
      expect(addListener.mock.calls.filter(([event]) => event === "tldw:open-command-palette")).toHaveLength(1)
      expect(removeListener.mock.calls.filter(([event]) => event === "tldw:open-command-palette")).toHaveLength(0)

      fireEvent.keyDown(input, { key: "Escape" })
      await waitFor(() => expect(screen.queryByRole("dialog", { name: "Command Palette" })).not.toBeInTheDocument())
      await waitFor(() => expect(trigger).toHaveFocus())
    })

    it.each([true, false])("keeps the existing page-help availability when hideHeader=%s", async (hidden) => {
      const view = render(tree(false, false))
      view.rerender(tree(hidden))
      await waitFor(() => expect(Boolean(screen.queryByTestId("shell-header"))).toBe(!hidden))
      act(() => window.dispatchEvent(new CustomEvent("tldw:open-help-modal")))
      if (name === "SharedLayout" || hidden) {
        expect(await screen.findByRole("dialog", { name: "Page help" })).toBeInTheDocument()
        expect(screen.getAllByRole("dialog", { name: "Page help" })).toHaveLength(1)
      } else {
        expect(screen.queryByRole("dialog", { name: "Page help" })).not.toBeInTheDocument()
      }
    })
  })
}

it("keeps the shared layout's current prompt commands after nested header changes", async () => {
  const previousAction = vi.fn()
  const currentAction = vi.fn()
  promptState.commands = [{ id: "prompt-current", label: "Current prompt", icon: null, action: previousAction, category: "prompt" }]
  const tree = (hidden: boolean) => <MemoryRouter><SharedLayout><SharedLayout hideHeader={hidden}><div>Route</div></SharedLayout></SharedLayout></MemoryRouter>
  const view = render(tree(false))
  act(() => window.dispatchEvent(new CustomEvent("tldw:open-command-palette")))
  const input = await screen.findByRole("textbox", { name: "Search commands" })
  await waitFor(() => expect(input).toHaveFocus())
  fireEvent.change(input, { target: { value: "Current prompt" } })
  promptState.commands = [{ id: "prompt-current", label: "Current prompt", icon: null, action: currentAction, category: "prompt" }]
  view.rerender(tree(true))
  await waitFor(() => expect(screen.queryByTestId("shell-header")).not.toBeInTheDocument())
  expect(input).toHaveValue("Current prompt")
  const command = screen.getAllByRole("option", { name: "Current prompt" })
  expect(command).toHaveLength(1)
  fireEvent.click(command[0])
  expect(currentAction).toHaveBeenCalledTimes(1)
  expect(previousAction).not.toHaveBeenCalled()
})
