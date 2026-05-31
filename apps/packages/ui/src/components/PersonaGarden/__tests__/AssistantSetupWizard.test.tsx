import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { AssistantSetupWizard } from "../AssistantSetupWizard"

describe("AssistantSetupWizard", () => {
  it("renders the persona-choice step first", () => {
    render(
      <AssistantSetupWizard
        catalog={[
          { id: "default_persona", name: "Default Persona" },
          { id: "helper", name: "Helper" }
        ]}
        selectedPersonaId="default_persona"
        currentStep="persona"
        postSetupTargetTab="profiles"
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("assistant-setup-overlay")).toBeInTheDocument()
    expect(screen.getByText("Choose a persona")).toBeInTheDocument()
    expect(screen.getByText("Default Persona")).toBeInTheDocument()
    expect(screen.getByText("Helper")).toBeInTheDocument()
  })

  it("does not auto-select the existing default persona without an explicit click", () => {
    const onUsePersona = vi.fn()

    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="persona"
        postSetupTargetTab="profiles"
        saving={false}
        error={null}
        onUsePersona={onUsePersona}
        onCreatePersona={vi.fn()}
      />
    )

    expect(onUsePersona).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Use Default Persona persona" }))

    expect(onUsePersona).toHaveBeenCalledWith("default_persona")
  })

  it("lets keyboard users choose a persona with enter and space", () => {
    const onUsePersona = vi.fn()

    render(
      <AssistantSetupWizard
        catalog={[
          { id: "default_persona", name: "Default Persona" },
          { id: "helper", name: "Helper" }
        ]}
        selectedPersonaId="default_persona"
        currentStep="persona"
        postSetupTargetTab="profiles"
        saving={false}
        error={null}
        onUsePersona={onUsePersona}
        onCreatePersona={vi.fn()}
      />
    )

    const helperButton = screen.getByRole("button", { name: "Use Helper persona" })
    expect(helperButton).toHaveAttribute("aria-pressed", "false")

    fireEvent.keyDown(helperButton, { key: "Enter" })
    fireEvent.keyDown(helperButton, { key: " " })

    expect(onUsePersona).toHaveBeenCalledTimes(2)
    expect(onUsePersona).toHaveBeenNthCalledWith(1, "helper")
    expect(onUsePersona).toHaveBeenNthCalledWith(2, "helper")
  })

  it("allows creating a new persona from the setup flow", () => {
    const onCreatePersona = vi.fn()

    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="persona"
        postSetupTargetTab="profiles"
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={onCreatePersona}
      />
    )

    fireEvent.change(screen.getByPlaceholderText("New persona name"), {
      target: { value: "Garden Butler" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create new persona" }))

    expect(onCreatePersona).toHaveBeenCalledWith("Garden Butler")
  })

  it("renders a non-persona step placeholder without persona-choice actions", () => {
    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="voice"
        postSetupTargetTab="profiles"
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("voice")
    expect(screen.queryByRole("button", { name: "Use this persona" })).not.toBeInTheDocument()
  })

  it("renders injected voice-step content when the wizard advances past persona choice", () => {
    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="voice"
        postSetupTargetTab="profiles"
        voiceStepContent={<div data-testid="setup-voice-content">Voice step</div>}
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("setup-voice-content")).toHaveTextContent("Voice step")
    expect(screen.queryByText("Setup step")).not.toBeInTheDocument()
  })

  it("renders a progress rail with completed, current, and pending setup steps", () => {
    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="commands"
        postSetupTargetTab="profiles"
        progressItems={[
          {
            step: "persona",
            label: "Choose persona",
            status: "completed",
            summary: "Persona selected"
          },
          {
            step: "voice",
            label: "Voice defaults",
            status: "completed",
            summary: "Voice defaults saved"
          },
          {
            step: "commands",
            label: "Starter commands",
            status: "current",
            summary: "Starter commands selected"
          },
          {
            step: "safety",
            label: "Safety and connections",
            status: "pending",
            summary: null
          },
          {
            step: "test",
            label: "Test and finish",
            status: "pending",
            summary: null
          }
        ]}
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("assistant-setup-progress")).toBeInTheDocument()
    expect(screen.getByTestId("assistant-setup-progress-step-persona")).toHaveAttribute(
      "data-status",
      "completed"
    )
    expect(screen.getByTestId("assistant-setup-progress-step-voice")).toHaveAttribute(
      "data-status",
      "completed"
    )
    expect(screen.getByTestId("assistant-setup-progress-step-commands")).toHaveAttribute(
      "data-status",
      "current"
    )
    expect(screen.getByTestId("assistant-setup-progress-step-safety")).toHaveAttribute(
      "data-status",
      "pending"
    )
    expect(screen.getByTestId("assistant-setup-progress-step-test")).toHaveAttribute(
      "data-status",
      "pending"
    )
    expect(screen.getByText("Voice defaults saved")).toBeInTheDocument()
    expect(screen.getByText("Starter commands selected")).toBeInTheDocument()
  })

  it("renders optional compact visual setup without adding a required setup step", () => {
    const onOpenVisualSetup = vi.fn()

    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="voice"
        postSetupTargetTab="profiles"
        progressItems={[
          { step: "persona", label: "Choose persona", status: "completed", summary: null },
          { step: "voice", label: "Voice defaults", status: "current", summary: null }
        ]}
        visualSetupContent={
          <div data-testid="setup-visual-content">
            <button type="button" onClick={onOpenVisualSetup}>
              Set up visual buddy
            </button>
          </div>
        }
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("setup-visual-content")).toBeInTheDocument()
    expect(screen.queryByTestId("assistant-setup-progress-step-visuals")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /set up visual buddy/i }))
    expect(onOpenVisualSetup).toHaveBeenCalledTimes(1)
  })

  it("renders injected commands-step content when the wizard reaches starter commands", () => {
    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="commands"
        postSetupTargetTab="profiles"
        commandsStepContent={<div data-testid="setup-commands-content">Commands step</div>}
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("setup-commands-content")).toHaveTextContent("Commands step")
    expect(screen.queryByText("Setup step")).not.toBeInTheDocument()
  })

  it("renders injected safety-step content when the wizard reaches safety and connections", () => {
    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="safety"
        postSetupTargetTab="profiles"
        safetyStepContent={<div data-testid="setup-safety-content">Safety step</div>}
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("setup-safety-content")).toHaveTextContent("Safety step")
    expect(screen.queryByText("Setup step")).not.toBeInTheDocument()
  })

  it("renders injected test-step content when the wizard reaches finish setup", () => {
    render(
      <AssistantSetupWizard
        catalog={[{ id: "default_persona", name: "Default Persona" }]}
        selectedPersonaId="default_persona"
        currentStep="test"
        postSetupTargetTab="profiles"
        testStepContent={<div data-testid="setup-test-content">Test step</div>}
        saving={false}
        error={null}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(screen.getByTestId("setup-test-content")).toHaveTextContent("Test step")
    expect(screen.queryByText("Setup step")).not.toBeInTheDocument()
  })
})
