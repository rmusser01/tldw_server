// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useRouteTransitionStore } from "@/store/route-transition"
import {
  executeKeyboardShortcuts,
  useChatModeShortcuts,
  useModeNavigationShortcuts,
  useSidebarShortcuts,
  useWebSearchShortcuts
} from "../useKeyboardShortcuts"

const typeInto = () => {
  const textarea = document.createElement("textarea")
  document.body.appendChild(textarea)
  textarea.focus()
  return textarea
}

const pressFrom = (target: EventTarget, init: KeyboardEventInit) => {
  const event = new KeyboardEvent("keydown", {
    bubbles: true,
    cancelable: true,
    ...init
  })
  act(() => {
    target.dispatchEvent(event)
  })
  return event
}

describe("shortcuts pressed while typing", () => {
  beforeEach(() => {
    document.body.innerHTML = ""
    useRouteTransitionStore.getState().stop()
  })
  afterEach(() => {
    document.body.innerHTML = ""
  })

  it("leaves Ctrl+E to the text field instead of switching to chat with the current page (XS-08)", () => {
    const toggleChatMode = vi.fn()
    renderHook(() => useChatModeShortcuts(toggleChatMode))
    const textarea = typeInto()

    const event = pressFrom(textarea, { key: "e", code: "KeyE", ctrlKey: true })

    expect(toggleChatMode).not.toHaveBeenCalled()
    // macOS uses Ctrl+E for "move to end of line"; the browser must keep it.
    expect(event.defaultPrevented).toBe(false)
  })

  it("still toggles chat with the current page when focus is outside a text field (XS-08)", () => {
    const toggleChatMode = vi.fn()
    renderHook(() => useChatModeShortcuts(toggleChatMode))

    pressFrom(document.body, { key: "e", code: "KeyE", ctrlKey: true })

    expect(toggleChatMode).toHaveBeenCalledTimes(1)
  })

  it("leaves Ctrl/Cmd+B to the text field instead of hiding the sidebar mid-sentence (XP-07)", () => {
    const toggleSidebar = vi.fn()
    renderHook(() => useSidebarShortcuts(toggleSidebar))
    const textarea = typeInto()

    const event = pressFrom(textarea, { key: "b", code: "KeyB", ctrlKey: true })

    expect(toggleSidebar).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(false)
  })

  it("switches area with Alt+5 from the focused chat composer (XP-10)", () => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate))
    const textarea = typeInto()

    pressFrom(textarea, { key: "5", code: "Digit5", altKey: true })

    expect(navigate).toHaveBeenCalledWith("/notes")
  })

  it("switches area with Option+5 on macOS outside a text field, where the key reports '∞' (XP-10)", () => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate))

    const event = pressFrom(document.body, { key: "∞", code: "Digit5", altKey: true })

    expect(navigate).toHaveBeenCalledWith("/notes")
    expect(event.defaultPrevented).toBe(true)
  })

  // macOS layouts type characters with Option+digit: US "∞", German/Swiss
  // "[" (Option+5), Spanish "@" (Option+2). In a text field that is typing.
  it.each([
    ["∞", "Digit5"],
    ["[", "Digit5"],
    ["]", "Digit6"],
    ["{", "Digit8"],
    ["@", "Digit2"],
    ["#", "Digit3"]
  ])("leaves Option+digit that types %j to the text field", (key, code) => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate))
    const textarea = typeInto()

    const event = pressFrom(textarea, { key, code, altKey: true })

    expect(navigate).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(false)
  })

  it("leaves Option+W that types '∑' to the side panel composer instead of toggling web search", () => {
    const toggleWebSearch = vi.fn()
    renderHook(() => useWebSearchShortcuts(toggleWebSearch))
    const textarea = typeInto()

    const event = pressFrom(textarea, { key: "∑", code: "KeyW", altKey: true })

    expect(toggleWebSearch).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(false)
  })

  it("still toggles web search with Alt+W from the composer where the chord types no text", () => {
    const toggleWebSearch = vi.fn()
    renderHook(() => useWebSearchShortcuts(toggleWebSearch))
    const textarea = typeInto()

    pressFrom(textarea, { key: "w", code: "KeyW", altKey: true })

    expect(toggleWebSearch).toHaveBeenCalledTimes(1)
  })

  it("does not treat Alt with a number-pad digit as an area shortcut (XP-10)", () => {
    const navigate = vi.fn()
    renderHook(() => useModeNavigationShortcuts(navigate))
    const textarea = typeInto()

    pressFrom(textarea, { key: "5", code: "Numpad5", altKey: true })

    expect(navigate).not.toHaveBeenCalled()
  })
})

describe("physical key matching", () => {
  const event = (init: Partial<KeyboardEvent>) =>
    ({
      key: "",
      code: "",
      ctrlKey: false,
      altKey: false,
      shiftKey: false,
      metaKey: false,
      target: null,
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      ...init
    }) as unknown as KeyboardEvent

  it("matches Option+letter chords by physical key when macOS reports a symbol", () => {
    const action = vi.fn()
    executeKeyboardShortcuts(event({ key: "∑", code: "KeyW", altKey: true }), [
      { shortcut: { key: "w", altKey: true }, action }
    ])
    expect(action).toHaveBeenCalledTimes(1)
  })

  it("keeps Ctrl+letter chords on the typed letter so other layouts are not double-bound", () => {
    const action = vi.fn()
    // Dvorak: the physical B key types "x"; Ctrl+X must stay Cut.
    executeKeyboardShortcuts(event({ key: "x", code: "KeyB", ctrlKey: true }), [
      { shortcut: { key: "b", ctrlKey: true }, action }
    ])
    expect(action).not.toHaveBeenCalled()
  })
})
