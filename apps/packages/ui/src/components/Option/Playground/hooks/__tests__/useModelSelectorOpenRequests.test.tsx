// @vitest-environment jsdom
/**
 * AX-01: Mod+E opens the composer's model picker, but only for the composer.
 * It must not open the picker over a modal, settings drawer or help dialog
 * (their focus lock would trap it), must not repeat while the key is held,
 * and must not pull focus out of an already open picker's search.
 */
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { isMac } from "@/hooks/useKeyboardShortcuts"
import { useModelSelectorOpenRequests } from "../useModelSelectorOpenRequests"

const switchModel = { key: "e", metaKey: isMac, ctrlKey: !isMac }

const Composer = ({
  openModelApiSelector,
  pickerOpen = false,
  dialog = false
}: {
  openModelApiSelector: () => void
  pickerOpen?: boolean
  dialog?: boolean
}) => {
  const returnFocusSelectorRef = React.useRef<string | null>(null)
  useModelSelectorOpenRequests({ openModelApiSelector, returnFocusSelectorRef, pickerOpen })
  return (
    <>
      <textarea aria-label="Message" />
      <button type="button" data-testid="model-selector">Model</button>
      {pickerOpen ? <input aria-label="Search models" /> : null}
      {dialog ? (
        <div role="dialog" aria-modal="true" aria-label="Chat settings">
          <input aria-label="Temperature" />
        </div>
      ) : null}
    </>
  )
}

afterEach(cleanup)

describe("useModelSelectorOpenRequests Mod+E", () => {
  it("opens the picker from the composer", () => {
    const open = vi.fn()
    render(<Composer openModelApiSelector={open} />)
    const message = screen.getByRole("textbox", { name: "Message" })
    message.focus()

    expect(fireEvent.keyDown(message, switchModel)).toBe(false)
    expect(open).toHaveBeenCalledTimes(1)
  })

  it("leaves an open dialog alone", () => {
    const open = vi.fn()
    render(<Composer openModelApiSelector={open} dialog />)
    const field = screen.getByRole("textbox", { name: "Temperature" })
    field.focus()

    expect(fireEvent.keyDown(field, switchModel)).toBe(true)
    expect(open).not.toHaveBeenCalled()
    expect(field).toHaveFocus()
  })

  it("ignores the key repeating while it is held", () => {
    const open = vi.fn()
    render(<Composer openModelApiSelector={open} />)
    const message = screen.getByRole("textbox", { name: "Message" })
    message.focus()

    fireEvent.keyDown(message, { ...switchModel, repeat: true })
    expect(open).not.toHaveBeenCalled()
  })

  it("leaves focus in the open picker's search", () => {
    const open = vi.fn()
    render(<Composer openModelApiSelector={open} pickerOpen />)
    const search = screen.getByRole("textbox", { name: "Search models" })
    search.focus()

    fireEvent.keyDown(search, switchModel)
    expect(open).not.toHaveBeenCalled()
    expect(search).toHaveFocus()
  })
})
