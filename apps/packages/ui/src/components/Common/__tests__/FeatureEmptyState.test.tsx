import React from "react"
import { Archive } from "lucide-react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import FeatureEmptyState from "../FeatureEmptyState"

describe("FeatureEmptyState", () => {
  it("adapts legacy empty states to the canonical EmptyState primitive", () => {
    const { container } = render(
      <FeatureEmptyState
        className="library-empty"
        icon={Archive}
        iconClassName="text-primary"
        title="No items yet"
        description="Create your first item to continue."
        examples={["Upload a document", "Import a saved item"]}
      />
    )

    const root = container.querySelector('[data-ds-component="EmptyState"]')
    expect(root).toBeTruthy()
    expect(root?.className).toContain("bg-surface/90")
    expect(root?.className).toContain("border-border/80")
    expect(root?.className).toContain("library-empty")
    expect(
      screen.getByRole("heading", { name: "No items yet" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Create your first item to continue.")
    ).toBeInTheDocument()
    expect(screen.getByText("Upload a document")).toBeInTheDocument()
    expect(screen.getByText("Import a saved item")).toBeInTheDocument()
  })

  it("preserves legacy action labels, handlers, disabled states and title attributes", () => {
    const onPrimaryAction = vi.fn()
    const onSecondaryAction = vi.fn()

    render(
      <FeatureEmptyState
        title="No prompts"
        primaryActionLabel="Create prompt"
        secondaryActionLabel="Import prompt"
        onPrimaryAction={onPrimaryAction}
        onSecondaryAction={onSecondaryAction}
        secondaryDisabled
      />
    )

    const primary = screen.getByRole("button", { name: "Create prompt" })
    const secondary = screen.getByRole("button", { name: "Import prompt" })

    expect(primary).toHaveAttribute("title", "Create prompt")
    expect(secondary).toHaveAttribute("title", "Import prompt")
    expect(secondary).toBeDisabled()

    fireEvent.click(primary)
    fireEvent.click(secondary)

    expect(onPrimaryAction).toHaveBeenCalledTimes(1)
    expect(onSecondaryAction).not.toHaveBeenCalled()
  })
})
