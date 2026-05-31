import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen, within } from "@testing-library/react"
import { ExecutionPanel } from "../ExecutionPanel"
import { useWorkflowEditorStore } from "@/store/workflow-editor"

vi.mock("@/components/Common/Workflow", () => ({
  WorkflowRunInspector: () => <div data-testid="workflow-run-inspector" />
}))

const originalLoadRunInvestigation =
  useWorkflowEditorStore.getState().loadRunInvestigation

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  useWorkflowEditorStore.setState({
    nodes: [],
    edges: [],
    runId: null,
    status: "idle",
    nodeStates: {},
    pendingApproval: null,
    error: null,
    startedAt: null,
    completedAt: null,
    runInvestigation: null,
    runInvestigationLoading: false,
    runInvestigationError: null,
    loadRunInvestigation: originalLoadRunInvestigation
  })
})

describe("ExecutionPanel design-system alerts", () => {
  it("renders execution errors through the design-system Alert", () => {
    useWorkflowEditorStore.setState({
      status: "failed",
      error: "Workflow step failed",
      loadRunInvestigation: vi.fn().mockResolvedValue(undefined)
    })

    render(<ExecutionPanel />)

    const error = screen.getByText("Workflow step failed")
    const alert = error.closest('[data-ds-component="Alert"]')
    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement
    expect(alertEl).toHaveTextContent("Execution Error")
    expect(alertEl).toHaveTextContent("Workflow step failed")
  })

  it("renders diagnostics failures through the design-system Alert", () => {
    useWorkflowEditorStore.setState({
      status: "failed",
      runId: "run-1",
      runInvestigationError: "Diagnostics service unavailable",
      loadRunInvestigation: vi.fn().mockResolvedValue(undefined)
    })

    render(<ExecutionPanel />)

    const error = screen.getByText("Diagnostics service unavailable")
    const alert = error.closest('[data-ds-component="Alert"]')
    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement
    expect(alertEl).toHaveTextContent("Diagnostics unavailable")
    expect(
      within(alertEl).getByRole("button", { name: "Retry diagnostics" })
    ).toBeInTheDocument()
  })
})
