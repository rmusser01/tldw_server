import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import type { WorkProductTemplateId } from "@/workspace-templates/types"
import { WorkProductTemplateChooser } from "../StudioPane/WorkProductTemplateChooser"

const renderChooser = (options?: {
  selectedTemplateId?: WorkProductTemplateId
  selectedSourceCount?: number
  onSelectTemplate?: (templateId: WorkProductTemplateId) => void
  disabled?: boolean
}) =>
  render(
    <WorkProductTemplateChooser
      selectedTemplateId={options?.selectedTemplateId ?? "executive_brief"}
      selectedSourceCount={options?.selectedSourceCount ?? 1}
      onSelectTemplate={options?.onSelectTemplate ?? vi.fn()}
      disabled={options?.disabled}
    />
  )

describe("WorkProductTemplateChooser", () => {
  it("renders executive brief as an available work product", () => {
    renderChooser()

    expect(
      screen.getByRole("button", { name: /executive brief/i })
    ).toBeInTheDocument()
  })

  it("enables executive brief when selected sources meet the requirement", () => {
    renderChooser({ selectedSourceCount: 1 })

    const executiveBrief = screen.getByRole("button", {
      name: /executive brief/i
    })

    expect(executiveBrief).toBeEnabled()
    expect(executiveBrief).not.toHaveAttribute("aria-disabled", "true")
  })

  it("shows the decision-ready executive brief description", () => {
    renderChooser()

    expect(screen.getByText(/decision-ready summary/i)).toBeInTheDocument()
  })

  it("marks templates unavailable when source requirements are not met", () => {
    renderChooser({ selectedSourceCount: 0 })

    const executiveBrief = screen.getByRole("button", {
      name: /executive brief/i
    })

    expect(executiveBrief).toHaveAttribute("aria-disabled", "true")
    expect(executiveBrief).toBeDisabled()
  })

  it("selects executive brief when it is the actionable work product", async () => {
    const user = userEvent.setup()
    const onSelectTemplate = vi.fn()
    renderChooser({ selectedSourceCount: 1, onSelectTemplate })

    await user.click(screen.getByRole("button", { name: /executive brief/i }))

    expect(onSelectTemplate).toHaveBeenCalledWith("executive_brief")
  })

  it("hides planned roadmap templates from the end-user chooser", () => {
    renderChooser({ selectedSourceCount: 3 })

    expect(
      screen.queryByRole("button", { name: /research dossier/i })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /competitive market memo/i })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /technical project spec/i })
    ).not.toBeInTheDocument()
  })

  it("does not show planned status copy in the default end-user state", () => {
    renderChooser({ selectedSourceCount: 1 })

    expect(screen.queryByText(/planned/i)).not.toBeInTheDocument()
  })

  it("uses native disabled behavior while output generation is in flight", async () => {
    const user = userEvent.setup()
    const onSelectTemplate = vi.fn()
    renderChooser({
      selectedSourceCount: 1,
      disabled: true,
      onSelectTemplate
    })

    const executiveBrief = screen.getByRole("button", {
      name: /executive brief/i
    })

    expect(executiveBrief).toHaveAttribute("aria-disabled", "true")
    expect(executiveBrief).toBeDisabled()
    await user.click(executiveBrief)
    expect(onSelectTemplate).not.toHaveBeenCalled()
  })
})
