import React from "react"
import { ConfigProvider } from "antd"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { PageHelpModalHost } from "../PageHelpModalHost"
import { TutorialRunner } from "../TutorialRunner"
import { QuickChatHelperButton } from "../QuickChatHelper/QuickChatHelperButton"
import NotesManagerOverlays from "@/components/Notes/NotesManagerOverlays"
import { notesUiStorage } from "@/components/Notes/notes-manager-utils"
import { useQuickChatStore } from "@/store/quick-chat"
import { useTutorialStore } from "@/store/tutorials"
import { useStoreMessageOption } from "@/store/option"

// Real launchers, AntD Dialog lifecycle, source stores and Joyride. IO and jsdom
// geometry are declared doubles; H1/account boundary matrices live in focused tests.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string | { defaultValue: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, initial: unknown) => [initial, vi.fn()]
}))
vi.mock("@/hooks/useQuickChat", () => ({
  useQuickChat: () => ({
    messages: [],
    sendMessage: vi.fn(),
    cancelStream: vi.fn(),
    isStreaming: false,
    hasModel: false,
    activeModel: null,
    currentModel: null,
    modelOverride: null,
    setModelOverride: vi.fn()
  })
}))
vi.mock("@/hooks/useChatModelCatalog", () => ({
  EMPTY_CHAT_MODELS: [],
  useChatModelCatalog: () => ({
    data: [],
    isLoading: false,
    isError: false,
    refetch: vi.fn()
  })
}))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionPhase: () => "connected",
  useIsConnected: () => true
}))
vi.mock("@/components/Notes/KeywordPickerModal", () => ({
  default: () => null
}))
vi.mock("@/components/Notes/notes-manager-utils", () => ({
  NOTES_TUTORIAL_SHOWN_STORAGE_KEY: "notes-tutorial-shown",
  notesUiStorage: { remove: vi.fn(async () => {}) },
  toKeywordTestIdSegment: (value: string) => value
}))
beforeEach(() => {
  vi.mocked(notesUiStorage.remove).mockClear()
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible")
  useStoreMessageOption.getState().startNewChatDraft()
  useTutorialStore.getState().resetProgress()
  useQuickChatStore.getState().setIsOpen(false)
  useQuickChatStore.getState().setAssistantMode("browse_guides")
})
afterEach(() => {
  useTutorialStore.getState().resetProgress()
  useQuickChatStore.getState().setIsOpen(false)
  vi.restoreAllMocks()
})
// Capture the actual rendered React handler for delayed-event lifetime controls.
const retainedClick = (button: HTMLElement) => {
  const key = Object.keys(button).find((name) =>
    name.startsWith("__reactProps$")
  )
  if (!key) throw new Error("Rendered React button props are unavailable")
  return (button as unknown as Record<string, { onClick: () => void }>)[key]
    .onClick
}

it.each(["Close", "reopen", "unmount"])(
  "retained actual Help Start refuses %s",
  async (boundary) => {
    const mounted = render(
      <MemoryRouter initialEntries={["/chat"]}>
        <button onClick={() => useTutorialStore.getState().openHelpModal()}>
          Help opener
        </button>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    const opener = screen.getByRole("button", { name: "Help opener" })
    opener.focus()
    fireEvent.click(opener)
    const help = await screen.findByRole("dialog", { name: "Page Help" })
    const originalStart = retainedClick(
      within(help).getByRole("button", { name: "Start", exact: true })
    )
    if (boundary === "unmount") mounted.unmount()
    else {
      fireEvent.click(
        within(help).getByRole("button", { name: "Close", exact: true })
      )
      if (boundary === "reopen") {
        opener.focus()
        fireEvent.click(opener)
        await screen.findByRole("dialog", { name: "Page Help" })
      }
    }
    const helpOpen = useTutorialStore.getState().isHelpModalOpen
    act(originalStart)
    expect(useTutorialStore.getState().activeRun).toBeNull()
    expect(useTutorialStore.getState().isHelpModalOpen).toBe(helpOpen)
  }
)

it.each([false, true])(
  "retained Notes Restart refuses a closed dialog (reopen=%s)",
  async (reopen) => {
    const kw = {
      keywordSuggestionOptions: [],
      keywordSuggestionSelection: [],
      availableKeywords: [],
      sortedKeywordPickerOptions: [],
      recentKeywordPickerOptions: [],
      keywordPickerSelection: [],
      keywordManagerVisibleItems: [],
      keywordMergeTargetOptions: [],
      keywordNoteCountByKey: {},
      keywordPickerQuery: "",
      keywordManagerQuery: "",
      keywordPickerSortMode: "alphabetical",
      keywordRenameDraft: null,
      keywordMergeDraft: null
    }
    const imp = {
      pendingImportFiles: [],
      importModalOpen: false,
      importSubmitting: false,
      importDuplicateStrategy: "skip"
    }
    function Fixture() {
      const [open, setOpen] = React.useState(false)
      const props = {
        kw,
        imp,
        isOnline: true,
        shortcutHelpOpen: open,
        setShortcutHelpOpen: setOpen,
        t: (_key: string, options: { defaultValue: string }) =>
          options.defaultValue
      } as unknown as React.ComponentProps<typeof NotesManagerOverlays>
      return (
        <>
          <button onClick={() => setOpen(true)}>Open Notes help</button>
          <NotesManagerOverlays {...props} />
        </>
      )
    }
    render(
      <MemoryRouter initialEntries={["/notes"]}>
        <Fixture />
      </MemoryRouter>
    )
    fireEvent.click(screen.getByRole("button", { name: "Open Notes help" }))
    const originalStart = retainedClick(
      await screen.findByTestId("notes-restart-tutorial")
    )
    const dialog = screen.getByRole("dialog", { name: "Keyboard shortcuts" })
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Close", exact: true })
    )
    if (reopen)
      fireEvent.click(screen.getByRole("button", { name: "Open Notes help" }))
    act(originalStart)
    expect(useTutorialStore.getState().activeRun).toBeNull()
    expect(notesUiStorage.remove).not.toHaveBeenCalled()
  }
)

const geometry = (target: HTMLElement) =>
  vi
    .spyOn(target, "getBoundingClientRect")
    .mockReturnValue(new DOMRect(10, 10, 120, 44))
const exitTour = async (opener: HTMLElement) => {
  const skip = screen.getByRole("button", { name: "Skip tour" })
  skip.focus()
  fireEvent.click(skip)
  await waitFor(() =>
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument()
  )
  await waitFor(() => expect(opener).toHaveFocus())
}

it("warm actual Quick Helper transfers its original opener through removal and real Joyride Skip", async () => {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <QuickChatHelperButton variant="inline" />
      <button data-testid="model-selector">Model target</button>
      <textarea
        aria-label="Exact Quick launch draft"
        defaultValue={"Quick exact draft.  \nSecond line."}
      />
      <TutorialRunner />
    </MemoryRouter>
  )
  geometry(screen.getByRole("button", { name: "Model target" }))
  const opener = screen.getByTestId("quick-chat-helper-open-button")
  opener.focus()
  fireEvent.click(opener)
  const warm = await screen.findByRole("dialog")
  fireEvent.click(
    within(warm).getByRole("button", { name: "Close", exact: true })
  )
  await waitFor(() =>
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  )
  opener.focus()
  fireEvent.click(opener)
  const start = await screen.findByTestId(
    "quick-chat-guides-tutorial-action-playground-basics"
  )
  start.focus()
  fireEvent.click(start)
  expect(useTutorialStore.getState().activeRun?.focusReturn?.target).toBe(
    opener
  )
  const tour = await screen.findByRole("alertdialog", {
    name: "Choose a Model"
  })
  expect(
    screen.queryByTestId("quick-chat-guides-tutorials-section")
  ).not.toBeInTheDocument()
  await waitFor(() =>
    expect(tour).toContainElement(document.activeElement as HTMLElement)
  )
  await exitTour(opener)
  expect(
    screen.getByRole("textbox", { name: "Exact Quick launch draft" })
  ).toHaveValue("Quick exact draft.  \nSecond line.")
})

it("actual Notes dialog close completion preserves tour focus and returns to its original opener", async () => {
  const kw = {
    keywordSuggestionOptions: [],
    keywordSuggestionSelection: [],
    availableKeywords: [],
    sortedKeywordPickerOptions: [],
    recentKeywordPickerOptions: [],
    keywordPickerSelection: [],
    keywordManagerVisibleItems: [],
    keywordMergeTargetOptions: [],
    keywordNoteCountByKey: {},
    keywordPickerQuery: "",
    keywordManagerQuery: "",
    keywordPickerSortMode: "alphabetical",
    keywordRenameDraft: null,
    keywordMergeDraft: null
  }
  const imp = {
    pendingImportFiles: [],
    importModalOpen: false,
    importSubmitting: false,
    importDuplicateStrategy: "skip"
  }
  const t = (_key: string, options: { defaultValue: string }) =>
    options.defaultValue
  function Fixture() {
    const [open, setOpen] = React.useState(false)
    const props = {
      kw,
      imp,
      isOnline: true,
      shortcutHelpOpen: open,
      setShortcutHelpOpen: setOpen,
      t
    } as unknown as React.ComponentProps<typeof NotesManagerOverlays>
    return (
      <>
        <button onClick={() => setOpen(true)}>Notes help opener</button>
        <div data-testid="notes-list-region">Notes target</div>
        <textarea
          aria-label="Exact Notes launch draft"
          defaultValue="Exact Notes draft.  "
        />
        <NotesManagerOverlays {...props} />
        <TutorialRunner />
      </>
    )
  }
  // Motion disabled supplies actual Dialog close completion deterministically in jsdom.
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <MemoryRouter initialEntries={["/notes"]}>
        <Fixture />
      </MemoryRouter>
    </ConfigProvider>
  )
  geometry(screen.getByTestId("notes-list-region"))
  const opener = screen.getByRole("button", { name: "Notes help opener" })
  opener.focus()
  fireEvent.click(opener)
  const start = await screen.findByTestId("notes-restart-tutorial")
  start.focus()
  fireEvent.click(start)
  expect(useTutorialStore.getState().activeRun?.focusReturn?.target).toBe(
    opener
  )
  const tour = await screen.findByRole("alertdialog", { name: "Notes List" })
  await waitFor(() =>
    expect(
      screen.queryByTestId("notes-shortcuts-modal")
    ).not.toBeInTheDocument()
  )
  await waitFor(() =>
    expect(tour).toContainElement(document.activeElement as HTMLElement)
  )
  await exitTour(opener)
  expect(
    screen.getByRole("textbox", { name: "Exact Notes launch draft" })
  ).toHaveValue("Exact Notes draft.  ")
})
