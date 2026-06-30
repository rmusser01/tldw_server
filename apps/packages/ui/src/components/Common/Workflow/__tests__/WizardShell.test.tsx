import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkflowDefinition } from "@/types/workflows"

const mocks = vi.hoisted(() => ({
  setWorkflowStep: vi.fn(),
  cancelWorkflow: vi.fn(),
  completeWorkflow: vi.fn(),
  setError: vi.fn()
}))

const workflowStoreState = {
  activeWorkflow: {
    id: "wf-test",
    workflowId: "quick-save" as const,
    status: "active" as const,
    currentStepIndex: 0,
    startedAt: 1,
    data: {}
  },
  isProcessing: false,
  processingProgress: 0,
  processingMessage: "",
  error: "The workflow failed",
  setWorkflowStep: mocks.setWorkflowStep,
  cancelWorkflow: mocks.cancelWorkflow,
  completeWorkflow: mocks.completeWorkflow,
  setError: mocks.setError
}

vi.mock("@/store/workflows", () => ({
  useWorkflowsStore: (selector: (state: typeof workflowStoreState) => unknown) =>
    selector(workflowStoreState)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      return _key
    }
  })
}))

import { WizardShell } from "../WizardShell"

const workflow: WorkflowDefinition = {
  id: "quick-save",
  category: "content-capture",
  labelToken: "Quick Save",
  descriptionToken: "Save the current page",
  icon: "save",
  steps: [
    {
      id: "details",
      labelToken: "Details",
      component: "DetailsStep"
    },
    {
      id: "review",
      labelToken: "Review",
      component: "ReviewStep"
    }
  ]
}

describe("WizardShell", () => {
  beforeEach(() => {
    workflowStoreState.error = "The workflow failed"
    workflowStoreState.isProcessing = false
    workflowStoreState.processingProgress = 0
    workflowStoreState.processingMessage = ""
    workflowStoreState.activeWorkflow.currentStepIndex = 0
    mocks.setWorkflowStep.mockReset()
    mocks.cancelWorkflow.mockReset()
    mocks.completeWorkflow.mockReset()
    mocks.setError.mockReset()
  })

  it("renders workflow errors with the design-system Alert and dismisses them", async () => {
    const user = userEvent.setup()

    render(
      <WizardShell workflow={workflow}>
        <div>Step body</div>
      </WizardShell>
    )

    expect(
      screen.getByText("Error").closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByText("The workflow failed")).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: /dismiss workflow error/i })
    )

    expect(mocks.setError).toHaveBeenCalledWith(null)
  })
})
