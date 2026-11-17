// @vitest-environment jsdom
import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import {
  useKeyboardShortcuts,
  type KeyboardShortcutConfig
} from "../useKeyboardShortcuts"

function CommittedControl({
  shortcuts,
  keyToPress
}: {
  shortcuts: KeyboardShortcutConfig[]
  keyToPress?: string
}) {
  useKeyboardShortcuts(shortcuts)
  React.useLayoutEffect(() => {
    if (keyToPress)
      document.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: keyToPress,
          ctrlKey: true,
          bubbles: true,
          cancelable: true
        })
      )
  }, [keyToPress])
  return (
    <span>{shortcuts[0]?.enabled === false ? "Unavailable" : "Available"}</span>
  )
}

const command = (
  action: () => void,
  key = "b",
  enabled = true
): KeyboardShortcutConfig[] => [
  { shortcut: { key, ctrlKey: true }, action, enabled }
]

describe("committed shortcut configuration", () => {
  it("stops the formerly enabled command before the disabled UI can receive a key", () => {
    const action = vi.fn()
    const view = render(<CommittedControl shortcuts={command(action)} />)
    view.rerender(
      <CommittedControl
        shortcuts={command(action, "b", false)}
        keyToPress="b"
      />
    )
    expect(action).not.toHaveBeenCalled()
  })

  it("performs the advertised rebound key with the current action before passive effects", () => {
    const previous = vi.fn(),
      current = vi.fn()
    const view = render(<CommittedControl shortcuts={command(previous)} />)
    view.rerender(
      <CommittedControl shortcuts={command(current, "n")} keyToPress="n" />
    )
    expect(current).toHaveBeenCalledOnce()
    expect(previous).not.toHaveBeenCalled()
  })

  it("uses the current callback for an unchanged available binding", () => {
    const previous = vi.fn(),
      current = vi.fn()
    const view = render(<CommittedControl shortcuts={command(previous)} />)
    view.rerender(
      <CommittedControl shortcuts={command(current)} keyToPress="b" />
    )
    expect(current).toHaveBeenCalledOnce()
    expect(previous).not.toHaveBeenCalled()
  })
})
