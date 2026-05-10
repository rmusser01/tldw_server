import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { VisualPackReusePanel } from "../VisualPackReusePanel"

const baseProps = {
  selectedPersonaName: "Research Buddy",
  hasSelectedPack: true,
  libraryItemCount: 2,
  hasDuplicateTargets: true,
  duplicateTargetsLoading: false,
  onCreateDraft: vi.fn(),
  onOpenLibrary: vi.fn(),
  onOpenImport: vi.fn(),
  onOpenDuplicate: vi.fn()
}

describe("VisualPackReusePanel", () => {
  it("routes user-owned visual pack reuse actions through existing flows", () => {
    const onCreateDraft = vi.fn()
    const onOpenLibrary = vi.fn()
    const onOpenImport = vi.fn()
    const onOpenDuplicate = vi.fn()

    render(
      <VisualPackReusePanel
        {...baseProps}
        onCreateDraft={onCreateDraft}
        onOpenLibrary={onOpenLibrary}
        onOpenImport={onOpenImport}
        onOpenDuplicate={onOpenDuplicate}
      />
    )

    const panel = screen.getByTestId("persona-visual-reuse-panel")
    expect(panel).toHaveTextContent("Research Buddy")
    expect(panel).toHaveTextContent("user-owned")
    expect(panel).toHaveTextContent("draft")
    expect(panel).toHaveTextContent("review")
    expect(panel).toHaveTextContent("activate")
    expect(panel).not.toHaveTextContent(/marketplace/i)
    expect(panel).not.toHaveTextContent(/shared with other users/i)
    expect(panel).not.toHaveTextContent(/\bVN\b/)
    expect(panel).not.toHaveTextContent(/CYOA/i)

    fireEvent.click(screen.getByRole("button", { name: /create draft/i }))
    fireEvent.click(screen.getByRole("button", { name: /use personal library/i }))
    fireEvent.click(screen.getByRole("button", { name: /import archive/i }))
    fireEvent.click(screen.getByRole("button", { name: /duplicate to persona/i }))

    expect(onCreateDraft).toHaveBeenCalledTimes(1)
    expect(onOpenLibrary).toHaveBeenCalledTimes(1)
    expect(onOpenImport).toHaveBeenCalledTimes(1)
    expect(onOpenDuplicate).toHaveBeenCalledTimes(1)
  })

  it("keeps unavailable duplicate flows disabled while leaving library discovery open", () => {
    const onOpenLibrary = vi.fn()
    const onOpenDuplicate = vi.fn()

    const { rerender } = render(
      <VisualPackReusePanel
        {...baseProps}
        hasSelectedPack={false}
        libraryItemCount={0}
        hasDuplicateTargets={false}
        onOpenLibrary={onOpenLibrary}
        onOpenDuplicate={onOpenDuplicate}
      />
    )

    expect(screen.getByTestId("persona-visual-reuse-panel")).toHaveTextContent(
      "No saved visual packs yet"
    )
    fireEvent.click(screen.getByRole("button", { name: /use personal library/i }))
    expect(onOpenLibrary).toHaveBeenCalledTimes(1)

    const duplicateButton = screen.getByRole("button", {
      name: /duplicate to persona/i
    })
    expect(duplicateButton).toBeDisabled()
    expect(screen.getByTestId("persona-visual-reuse-panel")).toHaveTextContent(
      "Select a pack before duplicating"
    )
    fireEvent.click(duplicateButton)
    expect(onOpenDuplicate).not.toHaveBeenCalled()

    rerender(
      <VisualPackReusePanel
        {...baseProps}
        hasSelectedPack
        hasDuplicateTargets={false}
        duplicateTargetsLoading={false}
        onOpenLibrary={onOpenLibrary}
        onOpenDuplicate={onOpenDuplicate}
      />
    )

    expect(screen.getByRole("button", { name: /duplicate to persona/i })).toBeDisabled()
    expect(screen.getByTestId("persona-visual-reuse-panel")).toHaveTextContent(
      "Add another persona before duplicating"
    )

    rerender(
      <VisualPackReusePanel
        {...baseProps}
        hasSelectedPack
        hasDuplicateTargets={false}
        duplicateTargetsLoading
        onOpenLibrary={onOpenLibrary}
        onOpenDuplicate={onOpenDuplicate}
      />
    )

    expect(screen.getByRole("button", { name: /duplicate to persona/i })).toBeDisabled()
  })
})
