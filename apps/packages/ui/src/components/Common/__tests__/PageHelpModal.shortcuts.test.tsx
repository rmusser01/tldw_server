import React from "react"
import { describe, expect, it, vi } from "vitest"
import {
  act,
  fireEvent,
  render,
  screen,
  within,
  waitFor
} from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { PageHelpModalHost } from "../PageHelpModalHost"
import { CommandPaletteHost } from "../CommandPaletteHost"
import { useTutorialStore } from "@/store/tutorials"
import { useChatShortcuts } from "@/hooks/keyboard/useKeyboardShortcuts"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  defaultShortcuts: {
    focusTextarea: { key: "Escape", shiftKey: true },
    newChat: { key: "u", ctrlKey: true, shiftKey: true },
    toggleSidebar: { key: "b", ctrlKey: true },
    toggleChatMode: { key: "e", ctrlKey: true },
    toggleWebSearch: { key: "w", altKey: true },
    toggleQuickChatHelper: { key: "h", ctrlKey: true, shiftKey: true },
    modePlayground: { key: "1", altKey: true },
    modeSources: { key: "2", altKey: true },
    modeMedia: { key: "3", altKey: true },
    modeKnowledge: { key: "4", altKey: true },
    modeNotes: { key: "5", altKey: true },
    modePrompts: { key: "6", altKey: true },
    modeFlashcards: { key: "7", altKey: true },
    modeWorldBooks: { key: "8", altKey: true },
    modeDictionaries: { key: "9", altKey: true },
    modeCharacters: { key: "0", altKey: true }
  },
  useShortcutConfig: () => ({
    shortcuts: {
      focusTextarea: { key: "Escape", shiftKey: true },
      newChat: { key: "x", ctrlKey: true, altKey: true },
      toggleSidebar: { key: "b", ctrlKey: true },
      toggleChatMode: { key: "e", ctrlKey: true },
      toggleWebSearch: { key: "w", altKey: true },
      toggleQuickChatHelper: { key: "h", ctrlKey: true, shiftKey: true },
      modePlayground: { key: "1", altKey: true },
      modeSources: { key: "2", altKey: true },
      modeMedia: { key: "3", altKey: true },
      modeKnowledge: { key: "4", altKey: true },
      modeNotes: { key: "5", altKey: true },
      modePrompts: { key: "6", altKey: true },
      modeFlashcards: { key: "7", altKey: true },
      modeWorldBooks: { key: "8", altKey: true },
      modeDictionaries: { key: "9", altKey: true },
      modeCharacters: { key: "0", altKey: true }
    }
  }),
  formatShortcut: ({
    key,
    altKey,
    ctrlKey,
    shiftKey
  }: {
    key: string
    altKey?: boolean
    ctrlKey?: boolean
    shiftKey?: boolean
  }) =>
    [
      ...(ctrlKey ? ["Ctrl"] : []),
      ...(altKey ? ["Alt"] : []),
      ...(shiftKey ? ["Shift"] : []),
      key
    ].join(" + ")
}))

vi.mock("@/hooks/keyboard/useKeyboardShortcuts", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/hooks/keyboard/useKeyboardShortcuts")
  >()),
  isMac: false
}))

vi.mock("@/components/Option/Prompt/usePromptPaletteCommands", () => ({
  usePromptPaletteCommands: () => []
}))

vi.mock("@/tutorials", () => ({
  getTutorialsForRoute: () => []
}))

describe("PageHelpModal shortcuts", () => {
  it("ignores a retained closed settings dialog when opening and dismissing help", async () => {
    useTutorialStore.getState().closeHelpModal()
    function RetainedSettings() {
      const [open, setOpen] = React.useState(false)
      return (
        <>
          <button onClick={() => setOpen(true)}>Open settings</button>
          <div style={{ display: open ? "block" : "none" }}>
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Retained settings"
              data-testid="retained-settings">
              <button onClick={() => setOpen(false)}>Close settings</button>
            </div>
          </div>
        </>
      )
    }
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <RetainedSettings />
        <button>Reference trigger</button>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }))
    expect(
      screen.getByRole("dialog", { name: "Retained settings" })
    ).toBeVisible()
    fireEvent.keyDown(screen.getByRole("button", { name: "Close settings" }), {
      key: "?",
      shiftKey: true
    })
    expect(useTutorialStore.getState().isHelpModalOpen).toBe(false)
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }))
    expect(screen.getByTestId("retained-settings").isConnected).toBe(true)
    expect(screen.getByTestId("retained-settings")).not.toBeVisible()
    const trigger = screen.getByRole("button", { name: "Reference trigger" })
    trigger.focus()
    fireEvent.keyDown(trigger, { key: "?", shiftKey: true })
    const help = await screen.findByRole("dialog", { name: "Page Help" })
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    fireEvent.keyDown(help, { key: "Escape" })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it("dismisses only the command palette on Escape over an open help reference", async () => {
    useTutorialStore.getState().closeHelpModal()
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <button>Reference trigger</button>
        <textarea aria-label="Retained draft" defaultValue="Exact draft?" />
        <PageHelpModalHost />
        <CommandPaletteHost />
      </MemoryRouter>
    )
    const trigger = screen.getByRole("button", { name: "Reference trigger" })
    trigger.focus()
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    const help = await screen.findByRole("dialog", { name: "Page Help" })
    const closeHelp = within(help).getByRole("button", { name: "Close" })
    await waitFor(() => expect(closeHelp).toHaveFocus())
    fireEvent.keyDown(closeHelp, { key: "k", ctrlKey: true })
    const paletteInput = await screen.findByRole("textbox", {
      name: "Search commands"
    })
    await waitFor(() => expect(paletteInput).toHaveFocus())
    fireEvent.keyDown(paletteInput, { key: "Escape" })
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Search commands" })
      ).not.toBeInTheDocument()
    )
    expect(help).toBeVisible()
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    await waitFor(() => expect(closeHelp).toHaveFocus())
    fireEvent.keyDown(closeHelp, { key: "Escape" })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
    expect(screen.getByRole("textbox", { name: "Retained draft" })).toHaveValue(
      "Exact draft?"
    )
  })

  it("restores the initiator after both cold and cached modal openings", async () => {
    useTutorialStore.getState().closeHelpModal()
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <button>Reference trigger</button>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    const trigger = screen.getByRole("button", { name: "Reference trigger" })
    for (const attempt of [0, 1]) {
      trigger.focus()
      act(() => window.dispatchEvent(new CustomEvent("tldw:open-help-modal")))
      const dialog = await screen.findByRole("dialog", { name: "Page Help" })
      fireEvent.click(within(dialog).getByRole("button", { name: "Close" }))
      expect(trigger, `open ${attempt}`).toHaveFocus()
    }
  })

  it("documents the effective configured chat binding and closes with focus restored", async () => {
    useTutorialStore.getState().closeHelpModal()
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <button>Reference trigger</button>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    const trigger = screen.getByRole("button", { name: "Reference trigger" })
    trigger.focus()
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    const dialog = await screen.findByRole("dialog", { name: "Page Help" })
    const newChat = within(dialog).getByText("Start new chat").parentElement!
    expect(within(newChat).getByText("Ctrl + Alt + x")).toBeVisible()
    expect(
      within(dialog).queryByText("Ctrl + Shift + u")
    ).not.toBeInTheDocument()
    fireEvent.keyDown(dialog, { key: "Escape" })
    expect(trigger).toHaveFocus()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("uses the same configured New binding as the actual keyboard handler", async () => {
    const onNew = vi.fn()
    function Bindings() {
      useChatShortcuts(onNew, true)
      return null
    }
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <Bindings />
        <button>Reference trigger</button>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    const trigger = screen.getByRole("button", { name: "Reference trigger" })
    trigger.focus()
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    const dialog = await screen.findByRole("dialog", { name: "Page Help" })
    expect(
      within(
        within(dialog).getByText("Start new chat").parentElement!
      ).getByText("Ctrl + Alt + x")
    ).toBeVisible()
    fireEvent.keyDown(dialog, { key: "Escape" })
    fireEvent.keyDown(trigger, { key: "u", ctrlKey: true, shiftKey: true })
    expect(onNew).not.toHaveBeenCalled()
    fireEvent.keyDown(trigger, { key: "x", ctrlKey: true, altKey: true })
    expect(onNew).toHaveBeenCalledOnce()
  })

  it("shows compact bindings without advertising full-app-only handlers", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModalHost scope="sidepanel" />
      </MemoryRouter>
    )
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    const dialog = await screen.findByRole("dialog", { name: "Page Help" })
    for (const text of [
      "Start new chat",
      "Toggle sidebar",
      "Switch normal / RAG chat mode",
      "Toggle web search",
      "Fork from the focused message"
    ])
      expect(within(dialog).getByText(text)).toBeVisible()
    for (const text of [
      "Toggle Quick Chat Helper",
      "Go to Sources",
      "Search this conversation",
      "Open mode launcher"
    ])
      expect(within(dialog).queryByText(text)).not.toBeInTheDocument()
  })

  it("omits page navigation bindings when the shell disables that handler", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModalHost modeNavigationEnabled={false} />
      </MemoryRouter>
    )
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )
    const dialog = await screen.findByRole("dialog", { name: "Page Help" })
    expect(within(dialog).getByText("Toggle sidebar")).toBeVisible()
    expect(within(dialog).queryByText("Go to Sources")).not.toBeInTheDocument()
  })

  it.each(["/chat", "/"])(
    "opens compact chat keyboard help at its %s entry",
    async (path) => {
      useTutorialStore.getState().closeHelpModal()
      render(
        <MemoryRouter initialEntries={[path]}>
          <button>Compact reference trigger</button>
          <PageHelpModalHost scope="sidepanel" />
        </MemoryRouter>
      )
      const trigger = screen.getByRole("button", {
        name: "Compact reference trigger"
      })
      trigger.focus()
      fireEvent.keyDown(trigger, { key: "?", shiftKey: true })
      const help = await screen.findByRole("dialog", { name: "Page Help" })
      expect(
        within(help).getByText("Switch normal / RAG chat mode")
      ).toBeVisible()
      expect(
        within(help).queryByText("Quick chat helper")
      ).not.toBeInTheDocument()
      fireEvent.keyDown(help, { key: "Escape" })
      expect(trigger).toHaveFocus()
    }
  )

  it("lists the Sources navigation shortcut when opened", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModalHost />
      </MemoryRouter>
    )

    act(() => window.dispatchEvent(new CustomEvent("tldw:open-help-modal")))

    expect(await screen.findByText("Go to Sources")).toBeVisible()
    expect(screen.getByText("Alt + 2")).toBeVisible()
  })
})
