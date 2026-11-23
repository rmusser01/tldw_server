// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { InlineUndoBanner, useInlineUndo, type InlineUndoOffer } from ".."

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

type HarnessApi = ReturnType<typeof useInlineUndo>

/**
 * A host that owns one undo offer, with a field to type in and a place for
 * focus to come back to, like the notes and chat lists that use the banner.
 */
const renderHost = () => {
  const api: { current: HarnessApi | null } = { current: null }
  const Host = () => {
    const undo = useInlineUndo()
    api.current = undo
    const listRef = React.useRef<HTMLDivElement>(null)
    return (
      <div>
        <input aria-label="Composer" />
        <div ref={listRef} tabIndex={-1} data-testid="list">
          list
        </div>
        <InlineUndoBanner undo={undo} onReturnFocus={() => listRef.current?.focus()} />
      </div>
    )
  }
  render(<Host />)
  const show = (offer: InlineUndoOffer) => act(() => api.current!.show(offer))
  return { api, show }
}

const offer = (overrides: Partial<InlineUndoOffer> = {}): InlineUndoOffer => ({
  key: "note-1",
  message: 'Moved "Plan" to Trash.',
  onUndo: vi.fn(),
  ...overrides
})

afterEach(() => {
  vi.useRealTimers()
})

// XP-17 and AX-06 (#3101, decision 8): a soft delete happens at once and
// offers Undo inline. The offer is announced, reachable by keyboard, and stays
// until the user dismisses it or leaves the page.
describe("InlineUndoBanner", () => {
  it("keeps an empty polite status region mounted before anything is deleted", () => {
    renderHost()

    const status = screen.getByRole("status")
    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveTextContent("")
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
  })

  it("announces what was deleted and describes Undo with it", () => {
    const { show } = renderHost()

    show(offer())

    expect(screen.getByRole("status")).toHaveTextContent('Moved "Plan" to Trash.')
    const undo = screen.getByRole("button", { name: "Undo" })
    expect(undo).toHaveAccessibleDescription('Moved "Plan" to Trash.')
    // The description is a quiet copy, so the message isn't read twice.
    const description = document.getElementById(undo.getAttribute("aria-describedby") ?? "")
    expect(description?.closest("[aria-live]")).toBeNull()
    expect(screen.getByRole("button", { name: "Dismiss" })).toBeInTheDocument()
  })

  it("does not expire on a timer", () => {
    vi.useFakeTimers()
    const { show } = renderHost()

    show(offer())
    act(() => {
      vi.advanceTimersByTime(10 * 60 * 1000)
    })

    expect(screen.getByRole("button", { name: "Undo" })).toBeInTheDocument()
  })

  it("moves focus to Undo when the delete left focus nowhere", async () => {
    const { show } = renderHost()
    ;(document.activeElement as HTMLElement | null)?.blur()

    show(offer())

    await waitFor(() => expect(screen.getByRole("button", { name: "Undo" })).toHaveFocus())
  })

  it("takes focus from the control that started the delete", async () => {
    const { show } = renderHost()
    const composer = screen.getByRole("textbox", { name: "Composer" })
    composer.focus()

    show(offer({ initiator: composer }))

    await waitFor(() => expect(screen.getByRole("button", { name: "Undo" })).toHaveFocus())
  })

  it("leaves focus on a list row the user moved to while the delete ran", async () => {
    const { show } = renderHost()
    const row = screen.getByTestId("list")
    row.focus()

    show(offer({ initiator: screen.getByRole("textbox", { name: "Composer" }) }))

    await act(async () => {})
    expect(row).toHaveFocus()
  })

  it("takes focus when the focused element is removed after the offer shows", async () => {
    const { show } = renderHost()
    const row = document.createElement("button")
    row.textContent = "Row menu"
    document.body.appendChild(row)
    row.focus()

    show(offer())
    await act(async () => {})
    expect(row).toHaveFocus()

    await act(async () => {
      row.remove()
    })
    await waitFor(() => expect(screen.getByRole("button", { name: "Undo" })).toHaveFocus())
  })

  it("leaves focus in a text field the user is typing in", async () => {
    const { show } = renderHost()
    const composer = screen.getByRole("textbox", { name: "Composer" })
    composer.focus()

    show(offer())

    await act(async () => {})
    expect(composer).toHaveFocus()
  })

  it("restores, closes, announces the restore and returns focus to the list", async () => {
    const onUndo = vi.fn().mockResolvedValue(undefined)
    const { show } = renderHost()
    show(offer({ onUndo, restoredMessage: "Note restored." }))

    const undoButton = screen.getByRole("button", { name: "Undo" })
    undoButton.focus()
    fireEvent.click(undoButton)

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
    )
    expect(onUndo).toHaveBeenCalledTimes(1)
    expect(screen.getByRole("status")).toHaveTextContent("Note restored.")
    expect(screen.getByTestId("list")).toHaveFocus()
  })

  it("runs Undo once while the restore is in flight", async () => {
    let finish!: () => void
    const onUndo = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve
        })
    )
    const { show } = renderHost()
    show(offer({ onUndo }))

    const undoButton = screen.getByRole("button", { name: "Undo" })
    fireEvent.click(undoButton)
    fireEvent.click(undoButton)

    expect(onUndo).toHaveBeenCalledTimes(1)
    expect(undoButton).toHaveAttribute("aria-disabled", "true")
    await act(async () => {
      finish()
    })
  })

  it.each([
    ["resolves false", () => Promise.resolve(false)],
    ["throws", () => Promise.reject(new Error("offline"))]
  ])("keeps the offer and says the restore failed when it %s", async (_label, onUndo) => {
    const { show } = renderHost()
    show(offer({ onUndo }))

    fireEvent.click(screen.getByRole("button", { name: "Undo" }))

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Couldn't restore. Try again.")
    )
    expect(screen.getByRole("button", { name: "Undo" })).not.toHaveAttribute("aria-disabled", "true")
    expect(screen.getByRole("status")).toHaveTextContent('Moved "Plan" to Trash.')
  })

  it("announces the outcome message an Undo returns", async () => {
    const { show } = renderHost()
    show(offer({ onUndo: async () => "Restored 1 of 2 notes.", restoredMessage: "Restored 2 notes." }))

    fireEvent.click(screen.getByRole("button", { name: "Undo" }))

    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Restored 1 of 2 notes."))
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
  })

  it("closes on Dismiss and on Escape without restoring", async () => {
    const onUndo = vi.fn()
    const { show } = renderHost()

    show(offer({ onUndo }))
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()

    show(offer({ onUndo }))
    fireEvent.keyDown(screen.getByRole("button", { name: "Undo" }), { key: "Escape" })
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
    expect(onUndo).not.toHaveBeenCalled()
  })

  it("shows only the newest offer", () => {
    const { show } = renderHost()

    show(offer({ key: "a", message: "Moved A to Trash." }))
    show(offer({ key: "b", message: "Moved B to Trash." }))

    expect(screen.getByRole("status")).toHaveTextContent("Moved B to Trash.")
    expect(screen.getAllByRole("button", { name: "Undo" })).toHaveLength(1)
  })

  it("retires an offer by key once the item is restored some other way", () => {
    const { api, show } = renderHost()
    show(offer({ key: "a" }))

    act(() => api.current!.dismiss("b"))
    expect(screen.getByRole("button", { name: "Undo" })).toBeInTheDocument()

    act(() => api.current!.dismiss("a"))
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
  })

  it("runs a second action next to Undo, then closes and returns focus", async () => {
    const onClick = vi.fn()
    const { show } = renderHost()

    show(offer({ secondaryAction: { label: "View Trash", onClick } }))
    const viewTrash = screen.getByRole("button", { name: "View Trash" })
    viewTrash.focus()
    fireEvent.click(viewTrash)

    expect(onClick).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId("list")).toHaveFocus())
  })

  it("leaves a follow-up offer the Undo showed in place of its own", async () => {
    const { api, show } = renderHost()
    show(
      offer({
        key: "a,b",
        message: "Moved 2 chats to Trash.",
        onUndo: async () => {
          api.current!.show(offer({ key: "b", message: "Restored 1 of 2 chats; 1 is still in Trash." }))
        }
      })
    )

    fireEvent.click(screen.getByRole("button", { name: "Undo" }))

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Restored 1 of 2 chats; 1 is still in Trash.")
    )
    expect(screen.getByRole("button", { name: "Undo" })).toBeInTheDocument()
  })

  it("keeps show and dismiss stable across offers", () => {
    const { api, show } = renderHost()
    const first = api.current!
    show(offer())
    expect(api.current!.show).toBe(first.show)
    expect(api.current!.dismiss).toBe(first.dismiss)
    expect(api.current!.current).toBe(first.current)
  })
})
