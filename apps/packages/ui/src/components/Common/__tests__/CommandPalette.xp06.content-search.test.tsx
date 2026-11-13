/**
 * XP-06 slice 1: ⌘K finds chats and notes.
 *
 * The palette searched only its own commands, so "asyncio" or "LoRA" found
 * nothing although matching chats and notes existed. Content providers now
 * add async "Chats" and "Notes" groups: debounced, cancelled when the query
 * changes, and at most five results each.
 */
import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter, useLocation } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  CommandPalette,
  type CommandItem,
  type CommandPaletteContentProvider
} from "../CommandPalette"
import { COMMAND_PALETTE_CONTENT_DEBOUNCE_MS } from "../command-palette-content"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({
    shortcuts: {},
    updateShortcut: vi.fn(),
    resetShortcuts: vi.fn(),
    resetShortcut: vi.fn()
  })
}))

const item = (category: "chat" | "note", id: string, action = vi.fn()): CommandItem => ({
  id: `${category}-${id}`,
  label: `${category === "chat" ? "Chat" : "Note"} ${id}`,
  icon: <span />,
  category,
  action,
  targetPath: category === "chat" ? `/chat?chat=${id}` : `/notes?note=${id}`
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

const renderPalette = async (providers: CommandPaletteContentProvider[]) => {
  render(
    <MemoryRouter>
      <CommandPalette contentProviders={providers} />
    </MemoryRouter>
  )
  window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))
  await screen.findByRole("dialog")
  return screen.getByRole("textbox", { name: "Search commands" })
}

const groupItems = (label: string) => {
  const heading = screen.getByText(label, { selector: "[data-command-group-label]" })
  const group = heading.parentElement as HTMLElement
  return within(group).getAllByRole("option")
}

describe("CommandPalette chats and notes (XP-06)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // jsdom has no scrollIntoView; the palette scrolls the highlighted row into view.
    Element.prototype.scrollIntoView = vi.fn()
  })

  it("searches chats and notes after a pause, five results each", async () => {
    const chats = vi.fn(async () =>
      Array.from({ length: 7 }, (_, index) => item("chat", `c${index}`))
    )
    const notes = vi.fn(async () => [item("note", "n1"), item("note", "n2")])
    const input = await renderPalette([
      { id: "chats", search: chats },
      { id: "notes", search: notes }
    ])

    fireEvent.change(input, { target: { value: "l" } })
    fireEvent.change(input, { target: { value: "lo" } })
    fireEvent.change(input, { target: { value: "lora" } })
    expect(chats).not.toHaveBeenCalled()

    await waitFor(() => expect(groupItems("Chats")).toHaveLength(5), {
      timeout: COMMAND_PALETTE_CONTENT_DEBOUNCE_MS + 1000
    })
    expect(chats).toHaveBeenCalledTimes(1)
    expect(chats).toHaveBeenCalledWith("lora", expect.any(AbortSignal))
    expect(notes).toHaveBeenCalledWith("lora", expect.any(AbortSignal))
    expect(groupItems("Notes").map((option) => option.textContent)).toEqual([
      expect.stringContaining("Note n1"),
      expect.stringContaining("Note n2")
    ])
    expect(groupItems("Chats")[0]).toHaveAttribute("data-target-path", "/chat?chat=c0")
  })

  it("cancels a search when the query changes", async () => {
    const first = deferred<CommandItem[]>()
    const signals: AbortSignal[] = []
    const chats = vi.fn((query: string, signal: AbortSignal) => {
      signals.push(signal)
      return query === "lora" ? first.promise : Promise.resolve([item("chat", "late")])
    })
    const input = await renderPalette([{ id: "chats", search: chats }])

    fireEvent.change(input, { target: { value: "lora" } })
    await waitFor(() => expect(chats).toHaveBeenCalledTimes(1), {
      timeout: COMMAND_PALETTE_CONTENT_DEBOUNCE_MS + 1000
    })
    fireEvent.change(input, { target: { value: "lorax" } })
    expect(signals[0].aborted).toBe(true)

    await act(async () => {
      first.resolve([item("chat", "stale")])
    })
    await waitFor(() => expect(groupItems("Chats")).toHaveLength(1), {
      timeout: COMMAND_PALETTE_CONTENT_DEBOUNCE_MS + 1000
    })
    expect(screen.queryByText("Chat stale")).not.toBeInTheDocument()
    expect(screen.getByText("Chat late")).toBeInTheDocument()
  })

  it("opens the highlighted result with Enter, in the order shown", async () => {
    const open = vi.fn()
    const notes = vi.fn(async () => [item("note", "n1", open)])
    const input = await renderPalette([{ id: "notes", search: notes }])

    fireEvent.change(input, { target: { value: "zzz-only-content" } })
    await waitFor(() => expect(groupItems("Notes")).toHaveLength(1), {
      timeout: COMMAND_PALETTE_CONTENT_DEBOUNCE_MS + 1000
    })
    const options = screen.getAllByRole("option")
    const selectedIndex = options.findIndex((option) => option.getAttribute("aria-selected") === "true")
    expect(options[selectedIndex]).toHaveAttribute("data-command-id", "note-n1")

    fireEvent.keyDown(input, { key: "Enter" })
    expect(open).toHaveBeenCalledTimes(1)
  })

  it("runs the command it highlights, in the order shown", async () => {
    const onNewChat = vi.fn()
    render(
      <MemoryRouter>
        <CommandPalette onNewChat={onNewChat} />
      </MemoryRouter>
    )
    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))
    await screen.findByRole("dialog")
    const input = screen.getByRole("textbox", { name: "Search commands" })

    fireEvent.keyDown(input, { key: "ArrowDown" })
    const options = screen.getAllByRole("option")
    expect(options[1]).toHaveAttribute("data-command-id", "action-new-chat")
    expect(options[1]).toHaveAttribute("aria-selected", "true")

    // Enter ran the second command in source order ("Go to Chat") instead.
    fireEvent.keyDown(input, { key: "Enter" })
    expect(onNewChat).toHaveBeenCalledTimes(1)
  })

  it("offers New note, which opens a new note on the Notes page", async () => {
    const Address = () => {
      const location = useLocation()
      return <output data-testid="address">{`${location.pathname}${location.search}`}</output>
    }
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <Address />
        <CommandPalette />
      </MemoryRouter>
    )
    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))
    await screen.findByRole("dialog")
    const input = screen.getByRole("textbox", { name: "Search commands" })

    fireEvent.change(input, { target: { value: "new note" } })
    const option = screen.getByRole("option", { name: /New note/ })
    expect(option).toHaveAttribute("data-command-id", "action-new-note")
    expect(option).toHaveAttribute("data-target-path", "/notes?new=1")
    expect(option).toHaveAttribute("aria-selected", "true")

    fireEvent.keyDown(input, { key: "Enter" })
    await waitFor(() => expect(screen.getByTestId("address")).toHaveTextContent("/notes?new=1"))
  })
})
