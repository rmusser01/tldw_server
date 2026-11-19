// @vitest-environment jsdom
/**
 * Decision [4] (#3101, CC-11+): in the web composer "@" searches notes, and
 * /note opens the same picker. Picking a note attaches its content as a
 * removable chip that names the note and links to it; the content goes out
 * with the message as pinned context does. The extension keeps "@" for tabs.
 */
import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const io = vi.hoisted(() => ({
  extension: false,
  searchNotes: vi.fn(async (_query: string, _params?: unknown, _options?: unknown) => ({
    items: [
      { id: 7, title: "Rebase notes", content: "Use git rebase -i.", snippet: [{ text: "Use git rebase" }] },
      { id: 8, title: "Release checklist", content: "Tag, build, publish." }
    ]
  }))
}))

vi.mock("@/utils/browser-runtime", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/browser-runtime")>()),
  isExtensionRuntime: () => io.extension
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { searchNotes: io.searchNotes }
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string; count?: number }, options?: Record<string, unknown>) => {
      const text = typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? _key)
      const values = { ...(typeof fallback === "object" ? fallback : {}), ...(options ?? {}) }
      return text.replace(/\{\{(\w+)\}\}/g, (_m, name) => String((values as Record<string, unknown>)[name] ?? ""))
    }
  })
}))

import { ComposerTextarea } from "../ComposerTextarea"

const props = (overrides: Record<string, unknown> = {}) => ({
  textareaRef: React.createRef<HTMLTextAreaElement>(),
  value: "",
  displayValue: "",
  onChange: vi.fn(),
  onKeyDown: vi.fn(),
  onPaste: vi.fn(),
  onFocus: vi.fn(),
  onSelect: vi.fn(),
  onCompositionStart: vi.fn(),
  onCompositionEnd: vi.fn(),
  onMouseDown: vi.fn(),
  onMouseUp: vi.fn(),
  placeholder: "Write a message…",
  isProMode: true,
  isMobile: false,
  isConnectionReady: true,
  isCollapsed: false,
  formInputProps: {},
  showSlashMenu: false,
  slashCommands: [],
  slashActiveIndex: 0,
  onSlashSelect: vi.fn(),
  onSlashActiveIndexChange: vi.fn(),
  slashEmptyLabel: "No results found",
  showMentions: false,
  filteredTabs: [],
  mentionPosition: null,
  onMentionSelect: vi.fn(),
  onMentionsClose: vi.fn(),
  onMentionRefetch: vi.fn(async () => undefined),
  onMentionsOpen: vi.fn(async () => undefined),
  draftSaved: false,
  onSetMessage: vi.fn(),
  ...overrides
})

const renderComposer = (overrides: Record<string, unknown> = {}) => {
  const all = props(overrides)
  const view = render(
    <MemoryRouter>
      <ComposerTextarea {...(all as any)} />
    </MemoryRouter>
  )
  return { ...view, props: all }
}

beforeEach(() => {
  io.extension = false
  io.searchNotes.mockClear()
  useStoreMessageOption.setState({ ragPinnedResults: [] })
})

describe("ComposerTextarea note mentions", () => {
  it("lists matching notes as the textarea's suggestions when @ is typed", async () => {
    renderComposer({ value: "Compare @reb", displayValue: "Compare @reb" })
    const listbox = await screen.findByRole("listbox", { name: "Notes" })
    expect(within(listbox).getByRole("option", { name: /Rebase notes/ })).toHaveAttribute("aria-selected", "true")
    await waitFor(() => expect(io.searchNotes).toHaveBeenCalledWith("reb", expect.anything(), expect.anything()))
    const textarea = screen.getByTestId("chat-input")
    expect(textarea).toHaveAttribute("aria-controls", listbox.id)
    expect(textarea.getAttribute("aria-activedescendant")).toBeTruthy()
  })

  it("attaches the picked note as pinned context and removes the @ text", async () => {
    const { props } = renderComposer({ value: "Compare @reb", displayValue: "Compare @reb" })
    await screen.findByRole("option", { name: /Rebase notes/ })
    const textarea = screen.getByTestId("chat-input")
    await act(async () => { fireEvent.keyDown(textarea, { key: "Enter" }) })
    expect(props.onKeyDown).not.toHaveBeenCalled()
    expect(props.onSetMessage).toHaveBeenCalledWith("Compare ")
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([
      expect.objectContaining({ id: "note:7", type: "note", title: "Rebase notes", snippet: "Use git rebase -i." })
    ])
  })

  it("shows an attached note as a removable chip that links to it", async () => {
    useStoreMessageOption.setState({
      ragPinnedResults: [{ id: "note:7", type: "note", title: "Rebase notes", source: "Notes", snippet: "Use git rebase -i." }]
    })
    renderComposer()
    const link = screen.getByRole("link", { name: "Rebase notes" })
    expect(link).toHaveAttribute("href", "/notes?note=7")
    fireEvent.click(screen.getByRole("button", { name: "Remove note Rebase notes" }))
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([])
  })

  it("moves through the notes with the arrow keys and closes on Escape", async () => {
    renderComposer({ value: "@", displayValue: "@" })
    await screen.findByRole("option", { name: /Release checklist/ })
    const textarea = screen.getByTestId("chat-input")
    fireEvent.keyDown(textarea, { key: "ArrowDown" })
    expect(screen.getByRole("option", { name: /Release checklist/ })).toHaveAttribute("aria-selected", "true")
    fireEvent.keyDown(textarea, { key: "Escape" })
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
  })

  it("opens the same picker for /note", async () => {
    io.extension = true
    renderComposer({ value: "/note release", displayValue: "/note release" })
    await screen.findByRole("listbox", { name: "Notes" })
    await waitFor(() => expect(io.searchNotes).toHaveBeenCalledWith("release", expect.anything(), expect.anything()))
  })

  it("keeps @ for tab mentions in the extension", async () => {
    io.extension = true
    renderComposer({ value: "Compare @reb", displayValue: "Compare @reb" })
    await act(async () => {})
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
    expect(io.searchNotes).not.toHaveBeenCalled()
  })
})
