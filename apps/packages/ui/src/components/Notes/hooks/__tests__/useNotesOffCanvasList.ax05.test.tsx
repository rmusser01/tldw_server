import React from "react"
import { createPortal } from "react-dom"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { useNotesOffCanvasList } from "../useNotesOffCanvasList"

// AX-05: the open mobile list traps Tab and closes on Escape, but only for
// keys pressed inside it. Popovers opened from the list render in a portal;
// React still bubbles their keys to the list, and those keys belong to them.

const OffCanvasList = ({ setMobileSidebarOpen }: { setMobileSidebarOpen: (open: boolean) => void }) => {
  const offCanvas = useNotesOffCanvasList({
    isMobileViewport: true,
    mobileSidebarOpen: true,
    sidebarCollapsed: false,
    setMobileSidebarOpen
  })
  return (
    <aside ref={offCanvas.asideRef} tabIndex={-1} onKeyDown={offCanvas.handleKeyDown} data-testid="list">
      <input aria-label="Search notes" data-notes-search-input="" />
      <button type="button">Last control</button>
      {createPortal(
        <div data-testid="popover">
          <button type="button">Popover first</button>
          <button type="button">Popover second</button>
        </div>,
        document.body
      )}
    </aside>
  )
}

describe("useNotesOffCanvasList AX-05 portaled popovers", () => {
  it("closes the list on Escape pressed inside it", () => {
    const setMobileSidebarOpen = vi.fn()
    render(<OffCanvasList setMobileSidebarOpen={setMobileSidebarOpen} />)

    fireEvent.keyDown(screen.getByLabelText("Search notes"), { key: "Escape" })
    expect(setMobileSidebarOpen).toHaveBeenCalledWith(false)
  })

  it("leaves Escape in a popover opened from the list to the popover", () => {
    const setMobileSidebarOpen = vi.fn()
    render(<OffCanvasList setMobileSidebarOpen={setMobileSidebarOpen} />)

    fireEvent.keyDown(screen.getByRole("button", { name: "Popover first" }), { key: "Escape" })
    expect(setMobileSidebarOpen).not.toHaveBeenCalled()
  })

  it("lets Tab move between controls in a popover opened from the list", () => {
    render(<OffCanvasList setMobileSidebarOpen={vi.fn()} />)

    const popoverButton = screen.getByRole("button", { name: "Popover second" })
    popoverButton.focus()
    const notCancelled = fireEvent.keyDown(popoverButton, { key: "Tab" })
    expect(notCancelled).toBe(true)
    expect(popoverButton).toHaveFocus()
  })
})
