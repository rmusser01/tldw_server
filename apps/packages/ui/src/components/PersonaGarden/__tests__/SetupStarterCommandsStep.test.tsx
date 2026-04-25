import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("../McpToolPicker", () => ({
  McpToolPicker: ({
    value,
    onChange,
    disabled
  }: {
    value: string
    onChange: (value: string) => void
    disabled?: boolean
  }) => (
    <input
      aria-label="MCP tool"
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
    />
  )
}))

import { SetupStarterCommandsStep } from "../SetupStarterCommandsStep"

describe("SetupStarterCommandsStep", () => {
  it("renders starter templates and an explicit skip action", () => {
    render(
      <SetupStarterCommandsStep
        saving={false}
        onCreateFromTemplate={vi.fn()}
        onCreateMcpStarter={vi.fn()}
        onSkip={vi.fn()}
      />
    )

    expect(screen.getByText("Starter commands")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Search Notes" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Note" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Search Library" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Continue without starter commands" })
    ).toBeInTheDocument()
  })

  it("creates a starter command from a shared template", () => {
    const onCreateFromTemplate = vi.fn()

    render(
      <SetupStarterCommandsStep
        saving={false}
        onCreateFromTemplate={onCreateFromTemplate}
        onCreateMcpStarter={vi.fn()}
        onSkip={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))

    expect(onCreateFromTemplate).toHaveBeenCalledWith("notes-search")
  })

  it("does not create a starter command again when an already-selected template is unchecked", () => {
    const onCreateFromTemplate = vi.fn()

    render(
      <SetupStarterCommandsStep
        saving={false}
        defaultCommands={[{ template_key: "notes-search" }]}
        onCreateFromTemplate={onCreateFromTemplate}
        onCreateMcpStarter={vi.fn()}
        onSkip={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))

    expect(onCreateFromTemplate).not.toHaveBeenCalled()
  })

  it("creates an MCP-backed starter command from an explicit tool and phrase", () => {
    const onCreateMcpStarter = vi.fn()

    render(
      <SetupStarterCommandsStep
        saving={false}
        onCreateFromTemplate={vi.fn()}
        onCreateMcpStarter={onCreateMcpStarter}
        onSkip={vi.fn()}
      />
    )

    fireEvent.change(screen.getByLabelText("MCP tool"), {
      target: { value: "notes.search" }
    })
    fireEvent.change(screen.getByPlaceholderText("Phrase users can say"), {
      target: { value: "search notes for {topic}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add MCP starter" }))

    expect(onCreateMcpStarter).toHaveBeenCalledWith(
      "notes.search",
      "search notes for {topic}"
    )
  })

  it("disables MCP starter inputs while saving and keeps the phrase input labeled", () => {
    render(
      <SetupStarterCommandsStep
        saving
        onCreateFromTemplate={vi.fn()}
        onCreateMcpStarter={vi.fn()}
        onSkip={vi.fn()}
      />
    )

    expect(screen.getByLabelText("MCP tool")).toBeDisabled()
    expect(screen.getByLabelText("MCP starter phrase")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Add MCP starter" })).toBeDisabled()
  })

  it("renders a step-local starter-command error while keeping retry actions available", () => {
    const onCreateFromTemplate = vi.fn()

    render(
      <SetupStarterCommandsStep
        saving={false}
        error="Failed to create starter command"
        onCreateFromTemplate={onCreateFromTemplate}
        onCreateMcpStarter={vi.fn()}
        onSkip={vi.fn()}
      />
    )

    expect(screen.getByText("Failed to create starter command")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))
    expect(onCreateFromTemplate).toHaveBeenCalledWith("notes-search")
  })

  it("shows retry guidance after starter-command creation fails while preserving skip", () => {
    const onSkip = vi.fn()

    render(
      <SetupStarterCommandsStep
        saving={false}
        error="Failed to create starter command"
        onCreateFromTemplate={vi.fn()}
        onCreateMcpStarter={vi.fn()}
        onSkip={onSkip}
      />
    )

    expect(
      screen.getByText(
        "Try a starter template again, add an MCP starter instead, or continue without starter commands."
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Continue without starter commands" }))
    expect(onSkip).toHaveBeenCalledTimes(1)
  })
})
