import { describe, it, expect, vi, afterEach } from "vitest"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import NodePalette from "../NodePalette"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { buildStepRegistry } from "../step-registry"

const originalLoadStepTypes = useWorkflowEditorStore.getState().loadStepTypes
const originalAddNode = useWorkflowEditorStore.getState().addNode

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  useWorkflowEditorStore.setState({
    stepRegistry: buildStepRegistry([]),
    stepTypesStatus: "idle",
    stepTypesError: null,
    loadStepTypes: originalLoadStepTypes,
    addNode: originalAddNode
  })
})

describe("NodePalette step-type loading UX", () => {
  it("shows degraded mode warning and retries loading", () => {
    const retryMock = vi.fn().mockResolvedValue(undefined)
    useWorkflowEditorStore.setState({
      stepRegistry: buildStepRegistry([]),
      stepTypesStatus: "error",
      stepTypesError: "Not Found",
      loadStepTypes: retryMock
    })

    render(<NodePalette />)

    const title = screen.getByText("Limited node library")
    expect(title).toBeInTheDocument()
    const alert = title.closest('[data-ds-component="Alert"]')
    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement
    expect(alertEl).toHaveTextContent(
      "Could not load server step types. Showing fallback steps."
    )
    fireEvent.click(within(alertEl).getByRole("button", { name: "Retry" }))
    expect(retryMock).toHaveBeenCalledWith(true)
  })

  it("shows retry control in empty error state", () => {
    const retryMock = vi.fn().mockResolvedValue(undefined)
    useWorkflowEditorStore.setState({
      stepRegistry: {} as ReturnType<typeof buildStepRegistry>,
      stepTypesStatus: "error",
      stepTypesError: "Not Found",
      loadStepTypes: retryMock
    })

    render(<NodePalette />)

    expect(screen.getByText("Unable to load step types")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry loading step types" }))
    expect(retryMock).toHaveBeenCalledWith(true)
  })

  it("adds a node when a palette item is clicked", () => {
    const addNodeMock = vi.fn().mockReturnValue("node-test")
    useWorkflowEditorStore.setState({
      stepRegistry: buildStepRegistry([]),
      stepTypesStatus: "ready",
      stepTypesError: null,
      nodes: [],
      zoom: 1,
      panPosition: { x: 0, y: 0 },
      addNode: addNodeMock
    })

    render(<NodePalette />)
    fireEvent.click(screen.getByRole("button", { name: "Add LLM Prompt" }))

    expect(addNodeMock).toHaveBeenCalledTimes(1)
    expect(addNodeMock).toHaveBeenCalledWith({
      type: "prompt",
      position: expect.objectContaining({
        x: expect.any(Number),
        y: expect.any(Number)
      })
    })
  })

  it("adds a node when pressing Enter on a palette item", () => {
    const addNodeMock = vi.fn().mockReturnValue("node-test")
    useWorkflowEditorStore.setState({
      stepRegistry: buildStepRegistry([]),
      stepTypesStatus: "ready",
      stepTypesError: null,
      nodes: [],
      zoom: 1,
      panPosition: { x: 0, y: 0 },
      addNode: addNodeMock
    })

    render(<NodePalette />)
    const item = screen.getByRole("button", { name: "Add LLM Prompt" })
    fireEvent.keyDown(item, { key: "Enter" })

    expect(addNodeMock).toHaveBeenCalledTimes(1)
  })

  it("matches alias terms in search", () => {
    useWorkflowEditorStore.setState({
      stepRegistry: buildStepRegistry([]),
      stepTypesStatus: "ready",
      stepTypesError: null
    })

    render(<NodePalette />)
    fireEvent.change(screen.getByPlaceholderText("Search nodes..."), {
      target: { value: "youtube" }
    })

    expect(screen.getByText("Media Ingest")).toBeInTheDocument()
  })
})
