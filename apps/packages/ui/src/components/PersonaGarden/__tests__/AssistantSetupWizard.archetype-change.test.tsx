import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { confirmMock } = vi.hoisted(() => ({
  confirmMock: vi.fn()
}))

vi.mock("antd", () => ({
  Modal: {
    confirm: confirmMock
  }
}))

vi.mock("../ArchetypePickerStep", () => ({
  ArchetypePickerStep: ({
    selectedKey,
    onSelect
  }: {
    selectedKey: string | null
    onSelect: (key: string) => void
  }) => (
    <div>
      <div data-testid="selected-archetype">{selectedKey ?? "none"}</div>
      <button type="button" onClick={() => onSelect("research_assistant")}>
        Select research assistant
      </button>
      <button type="button" onClick={() => onSelect("study_buddy")}>
        Select study buddy
      </button>
    </div>
  )
}))

import { Modal } from "antd"
import { AssistantSetupWizard } from "../AssistantSetupWizard"

describe("AssistantSetupWizard archetype changes", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    confirmMock.mockReset()
  })

  it("does not prompt just because the wizard enters the archetype step", () => {
    render(
      <AssistantSetupWizard
        catalog={[]}
        selectedPersonaId=""
        currentStep="archetype"
        postSetupTargetTab="profiles"
        archetypeKey="research_assistant"
        saving={false}
        error={null}
        onResetSetup={vi.fn()}
        onSelectArchetype={vi.fn()}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    expect(Modal.confirm).not.toHaveBeenCalled()
  })

  it("confirms before changing an existing archetype and only resets after approval", () => {
    const onResetSetup = vi.fn()
    const onSelectArchetype = vi.fn()

    render(
      <AssistantSetupWizard
        catalog={[]}
        selectedPersonaId=""
        currentStep="archetype"
        postSetupTargetTab="profiles"
        archetypeKey="research_assistant"
        saving={false}
        error={null}
        onResetSetup={onResetSetup}
        onSelectArchetype={onSelectArchetype}
        onUsePersona={vi.fn()}
        onCreatePersona={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Select study buddy" }))

    expect(Modal.confirm).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Change archetype?",
        content: "Changing your archetype will reset your customizations."
      })
    )
    const [firstDialog] = confirmMock.mock.calls[0]
    firstDialog.onCancel?.()
    expect(onResetSetup).not.toHaveBeenCalled()
    expect(onSelectArchetype).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Select study buddy" }))
    const [secondDialog] = confirmMock.mock.calls[1]
    secondDialog.onOk?.()

    expect(onResetSetup).toHaveBeenCalledTimes(1)
    expect(onSelectArchetype).toHaveBeenCalledWith("study_buddy")
  })
})
