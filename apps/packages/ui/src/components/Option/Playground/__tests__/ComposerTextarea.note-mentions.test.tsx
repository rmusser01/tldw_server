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
  hasNotes: true,
  getNote: vi.fn(async (noteId: string | number, _options?: unknown) => ({
    id: Number(noteId),
    title: "Rebase notes",
    content: "Full content from the server."
  })),
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
  tldwClient: { searchNotes: io.searchNotes, getNote: io.getNote }
}))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasNotes: io.hasNotes }, loading: false })
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

/** Puts the caret at the end, as typing there would. */
const typedAtEnd = () => {
  const textarea = screen.getByTestId("chat-input") as HTMLTextAreaElement
  act(() => {
    textarea.focus()
    textarea.setSelectionRange(textarea.value.length, textarea.value.length)
    fireEvent.select(textarea)
  })
  return textarea
}

beforeEach(() => {
  io.extension = false
  io.hasNotes = true
  io.getNote.mockClear()
  io.searchNotes.mockClear()
  useStoreMessageOption.setState({ ragPinnedResults: [] })
})

describe("ComposerTextarea note mentions", () => {
  it("lists matching notes as the textarea's suggestions when @ is typed", async () => {
    renderComposer({ value: "Compare @reb", displayValue: "Compare @reb" })
    typedAtEnd()
    const listbox = await screen.findByRole("listbox", { name: "Notes" })
    expect(within(listbox).getByRole("option", { name: /Rebase notes/ })).toHaveAttribute("aria-selected", "true")
    await waitFor(() => expect(io.searchNotes).toHaveBeenCalledWith("reb", { limit: 8, legacy_aliases: false, include_snippets: false }, expect.anything()))
    const textarea = screen.getByTestId("chat-input")
    expect(textarea).toHaveAttribute("aria-controls", listbox.id)
    expect(textarea.getAttribute("aria-activedescendant")).toBeTruthy()
  })

  it("attaches the picked note as pinned context and removes the @ text", async () => {
    const { props } = renderComposer({ value: "Compare @reb", displayValue: "Compare @reb" })
    const textarea = typedAtEnd()
    await screen.findByRole("option", { name: /Rebase notes/ })
    await act(async () => { fireEvent.keyDown(textarea, { key: "Enter" }) })
    expect(props.onKeyDown).not.toHaveBeenCalled()
    // The search row may be trimmed; the attached content is the note itself.
    await waitFor(() => expect(io.getNote).toHaveBeenCalledWith(7, expect.anything()))
    await waitFor(() => expect(props.onSetMessage).toHaveBeenCalledWith("Compare "))
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([
      expect.objectContaining({ id: "note:7", type: "note", title: "Rebase notes", snippet: "Full content from the server." })
    ])
  })

  it("shows an attached note as a removable chip that links to it", async () => {
    useStoreMessageOption.setState({
      ragPinnedResults: [{ id: "note:7", type: "note", title: "Rebase notes", source: "Notes", snippet: "Use git rebase -i." }]
    })
    renderComposer()
    const link = screen.getByRole("link", { name: "Rebase notes" })
    expect(link).toHaveAttribute("href", "/notes?note=7")
    const remove = screen.getByRole("button", { name: "Remove note Rebase notes" })
    // 32px on desktop, 44px on touch (lane 7A's size tokens replace these at integration).
    expect(remove.className).toMatch(/\bmin-h-8\b/)
    expect(remove.className).toMatch(/\[@media\(pointer:coarse\)\]:min-h-11/)
    fireEvent.click(remove)
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([])
    expect(screen.getByTestId("chat-input")).toHaveFocus()
  })

  it("moves through the notes with the arrow keys and closes on Escape", async () => {
    renderComposer({ value: "@", displayValue: "@" })
    const textarea = typedAtEnd()
    await screen.findByRole("option", { name: /Release checklist/ })
    fireEvent.keyDown(textarea, { key: "ArrowDown" })
    expect(screen.getByRole("option", { name: /Release checklist/ })).toHaveAttribute("aria-selected", "true")
    fireEvent.keyDown(textarea, { key: "Escape" })
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
  })

  it("opens the same picker for /note", async () => {
    io.extension = true
    renderComposer({ value: "/note release", displayValue: "/note release" })
    typedAtEnd()
    await screen.findByRole("listbox", { name: "Notes" })
    await waitFor(() => expect(io.searchNotes).toHaveBeenCalledWith("release", expect.anything(), expect.anything()))
  })

  it("keeps @ for tab mentions in the extension", async () => {
    io.extension = true
    renderComposer({ value: "Compare @reb", displayValue: "Compare @reb" })
    typedAtEnd()
    await act(async () => {})
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
    expect(io.searchNotes).not.toHaveBeenCalled()
  })

  it("does not attach the previous query's top note while a new search loads", async () => {
    const { props, rerender } = renderComposer({ value: "@reb", displayValue: "@reb" })
    typedAtEnd()
    await screen.findByRole("option", { name: /Rebase notes/ })
    io.searchNotes.mockImplementationOnce(() => new Promise(() => {}))
    rerender(<MemoryRouter><ComposerTextarea {...({ ...props, value: "@rel", displayValue: "@rel" } as any)} /></MemoryRouter>)
    const textarea = typedAtEnd()
    await waitFor(() => expect(screen.queryByRole("option", { name: /Rebase notes/ })).toBeNull())
    await waitFor(() => expect(io.searchNotes).toHaveBeenCalledWith("rel", expect.anything(), expect.anything()))
    await act(async () => { fireEvent.keyDown(textarea, { key: "Enter" }) })
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([])
    expect(io.getNote).not.toHaveBeenCalled()
  })

  it("searches again after the account changes", async () => {
    renderComposer({ value: "@reb", displayValue: "@reb" })
    typedAtEnd()
    await screen.findByRole("option", { name: /Rebase notes/ })
    const calls = io.searchNotes.mock.calls.length
    await act(async () => { window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")) })
    await waitFor(() => expect(io.searchNotes.mock.calls.length).toBeGreaterThan(calls))
  })

  it.each([
    ["Shift+Tab", { key: "Tab", shiftKey: true }],
    ["Ctrl+Enter", { key: "Enter", ctrlKey: true }],
    ["Cmd+Enter", { key: "Enter", metaKey: true }],
    ["an IME commit", { key: "Enter", keyCode: 229 }]
  ])("leaves %s to the composer", async (_name, init) => {
    const { props } = renderComposer({ value: "@reb", displayValue: "@reb" })
    const textarea = typedAtEnd()
    await screen.findByRole("option", { name: /Rebase notes/ })
    await act(async () => { fireEvent.keyDown(textarea, init) })
    expect(props.onKeyDown).toHaveBeenCalledOnce()
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([])
  })

  it("does not open for a draft set from code", async () => {
    renderComposer({ value: "@reb", displayValue: "@reb" })
    await act(async () => {})
    expect(screen.queryByText("Searching notes…")).toBeNull()
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
    expect(io.searchNotes).not.toHaveBeenCalled()
  })

  it("closes when the textarea loses focus", async () => {
    renderComposer({ value: "@reb", displayValue: "@reb" })
    const textarea = typedAtEnd()
    await screen.findByRole("listbox", { name: "Notes" })
    act(() => { fireEvent.blur(textarea) })
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
  })

  it("stays off while a long paste is collapsed", async () => {
    renderComposer({ value: "@reb", displayValue: "@reb", isCollapsed: true })
    typedAtEnd()
    await act(async () => {})
    expect(screen.queryByText("Searching notes…")).toBeNull()
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
  })

  it("shows only the note list while /note is open, not tab mentions", async () => {
    io.extension = true
    renderComposer({ value: "/note @x", displayValue: "/note @x", showMentions: true, filteredTabs: [{ id: 1, title: "Tab", url: "https://a.test" }] })
    typedAtEnd()
    await screen.findByRole("listbox", { name: "Notes" })
    expect(screen.queryByRole("listbox", { name: "Tab mention suggestions" })).toBeNull()
  })

  it("is off when the server has no Notes", async () => {
    io.hasNotes = false
    renderComposer({ value: "@reb", displayValue: "@reb" })
    typedAtEnd()
    await act(async () => {})
    expect(screen.queryByText("Searching notes…")).toBeNull()
    expect(screen.queryByRole("listbox", { name: "Notes" })).toBeNull()
  })
})
