import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { WorkflowEditor } from "../WorkflowEditor"

vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => false
}))

vi.mock("../WorkflowCanvas", () => ({
  WorkflowCanvas: () => <div data-testid="workflow-canvas">Canvas</div>
}))

vi.mock("../NodePalette", () => ({
  NodePalette: () => <div data-testid="panel-palette">Palette Panel</div>
}))

vi.mock("../NodeConfigPanel", () => ({
  NodeConfigPanel: () => <div data-testid="panel-config">Config Panel</div>
}))

vi.mock("../ExecutionPanel", () => ({
  ExecutionPanel: () => <div data-testid="panel-execution">Execution Panel</div>
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const originalState = {
  newWorkflow: useWorkflowEditorStore.getState().newWorkflow,
  loadStepTypes: useWorkflowEditorStore.getState().loadStepTypes,
  validate: useWorkflowEditorStore.getState().validate,
  canUndo: useWorkflowEditorStore.getState().canUndo,
  canRedo: useWorkflowEditorStore.getState().canRedo
}

beforeEach(() => {
  useWorkflowEditorStore.setState({
    workflowName: "Untitled Workflow",
    isDirty: false,
    nodes: [],
    edges: [],
    isMiniMapVisible: true,
    isGridVisible: true,
    sidebarPanel: "palette",
    issues: [],
    status: "idle",
    stepTypesStatus: "ready",
    newWorkflow: vi.fn(),
    loadStepTypes: vi.fn().mockResolvedValue(undefined),
    validate: vi.fn(() => ({ isValid: true, issues: [] })),
    canUndo: vi.fn(() => false),
    canRedo: vi.fn(() => false)
  })
})

afterEach(() => {
  cleanup()
  useWorkflowEditorStore.setState({
    newWorkflow: originalState.newWorkflow,
    loadStepTypes: originalState.loadStepTypes,
    validate: originalState.validate,
    canUndo: originalState.canUndo,
    canRedo: originalState.canRedo
  })
})

describe("WorkflowEditor responsive layout", () => {
  it("surfaces compact workflow readiness before the canvas", () => {
    useWorkflowEditorStore.setState({
      isDirty: true,
      stepTypesStatus: "ready",
      issues: [
        {
          id: "error-1",
          severity: "error",
          message: "Prompt node is missing a model"
        },
        {
          id: "warn-1",
          severity: "warning",
          message: "Workflow has no description"
        }
      ]
    })

    render(<WorkflowEditor />)

    const summary = screen.getByTestId("workflow-editor-status-summary")
    expect(summary).toHaveTextContent("Step types ready")
    expect(summary).toHaveTextContent("1 error")
    expect(summary).toHaveTextContent("1 warning")
    expect(summary).toHaveTextContent("Unsaved changes")
    expect(
      screen.getByRole("button", { name: "Open workflow panels" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Import workflow" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export workflow" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open run panel" })).toBeInTheDocument()
  })

  it("names workflow step-type loading and failure states", () => {
    useWorkflowEditorStore.setState({
      stepTypesStatus: "loading"
    })

    const { rerender } = render(<WorkflowEditor />)

    expect(screen.getByTestId("workflow-editor-status-summary")).toHaveTextContent(
      "Step types loading"
    )

    useWorkflowEditorStore.setState({
      stepTypesStatus: "error",
      stepTypesError: "Step type registry unavailable"
    })
    rerender(<WorkflowEditor />)

    expect(screen.getByTestId("workflow-editor-status-summary")).toHaveTextContent(
      "Step types unavailable"
    )
    expect(screen.getByTestId("workflow-editor-status-summary")).toHaveTextContent(
      "Step type registry unavailable"
    )
  })

  it("shows a named validation-ready state when there are no issues", () => {
    render(<WorkflowEditor />)

    expect(screen.getByTestId("workflow-editor-status-summary")).toHaveTextContent(
      "Validation clear"
    )
  })

  it("opens workflow panels in a drawer on non-desktop viewports", () => {
    render(<WorkflowEditor />)

    expect(screen.getByTestId("workflow-canvas")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open workflow panels" })
    ).toBeInTheDocument()
    expect(screen.queryByText("Workflow panels")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open workflow panels" }))

    expect(screen.getByText("Workflow panels")).toBeInTheDocument()
    expect(screen.getByTestId("panel-palette")).toBeInTheDocument()
  })

  it("includes validation issue counts in the icon-only control label", () => {
    useWorkflowEditorStore.setState({
      issues: [
        {
          id: "warn-1",
          severity: "warning",
          message: "Configuration warning"
        },
        {
          id: "warn-2",
          severity: "warning",
          message: "Another warning"
        }
      ]
    })

    render(<WorkflowEditor />)

    expect(
      screen.getByRole("button", {
        name: "Validation issues: 2 warnings"
      })
    ).toBeInTheDocument()
  })

  it("renders the selected non-default sidebar panel through its lazy boundary", async () => {
    useWorkflowEditorStore.setState({
      sidebarPanel: "config"
    })

    render(<WorkflowEditor />)

    fireEvent.click(screen.getByRole("button", { name: "Open workflow panels" }))
    expect(await screen.findByTestId("panel-config")).toBeInTheDocument()
    expect(screen.queryByTestId("panel-execution")).not.toBeInTheDocument()
  })
})
