/**
 * Workflow Editor Zustand Store
 *
 * Manages state for the node-based workflow editor including:
 * - Canvas state (nodes, edges)
 * - UI state (selection, panels, zoom)
 * - Execution state (run status, step states)
 * - History (undo/redo)
 *
 * @example
 * ```tsx
 * import { useWorkflowEditorStore } from "@/store/workflow-editor"
 *
 * // Add a node
 * const addNode = useWorkflowEditorStore((s) => s.addNode)
 * addNode({ type: "prompt", position: { x: 100, y: 100 } })
 *
 * // Get selected nodes
 * const selectedNodeIds = useWorkflowEditorStore((s) => s.selectedNodeIds)
 * ```
 */

import { createWithEqualityFn } from "zustand/traditional"
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge as rfAddEdge,
  type NodeChange,
  type EdgeChange,
  type Connection,
  type XYPosition
} from "@xyflow/react"
import type {
  WorkflowNode,
  WorkflowEdge,
  WorkflowNodeData,
  WorkflowStepType,
  WorkflowStepSchema,
  EditorUIState,
  EditorTool,
  SidebarPanel,
  WorkflowRunState,
  StepExecutionState,
  StepExecutionStatus,
  HumanApprovalRequest,
  HumanApprovalResponse,
  EditorHistoryEntry,
  WorkflowRunInvestigation,
  ServerWorkflowDefinition,
  ValidationIssue
} from "@/types/workflow-editor"
import {
  BASE_STEP_REGISTRY,
  buildStepRegistry,
  humanizeStepType,
  type StepRegistry
} from "@/components/WorkflowEditor/step-registry"
import {
  getWorkflowInvestigation,
  getWorkflowStepTypes
} from "@/services/tldw/workflows"

// ─────────────────────────────────────────────────────────────────────────────
// Storage Keys
// ─────────────────────────────────────────────────────────────────────────────

const EDITOR_STATE_KEY = "tldw:workflow-editor:state"
const RECENT_WORKFLOWS_KEY = "tldw:workflow-editor:recent"

// ─────────────────────────────────────────────────────────────────────────────
// State Types
// ─────────────────────────────────────────────────────────────────────────────

interface CanvasState {
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  workflowId: string | null
  workflowVersion: number | null
  savedCanvasKey: string | null
  ownerScopeKey: string | null
  viewRevision: number
  workflowName: string
  workflowDescription: string
  isDirty: boolean
}

interface UIState extends EditorUIState {
  isLoading: boolean
  error: string | null
}

interface ExecutionState extends WorkflowRunState {
  runWorkflowId: number | null
  runWorkflowVersion: number | null
  pendingRunSubmission: { key: string; idempotencyKey: string; inputs: Record<string, unknown> } | null
  isConnected: boolean
  runInvestigation: WorkflowRunInvestigation | null
  runInvestigationLoading: boolean
  runInvestigationError: string | null
}

interface HistoryState {
  past: EditorHistoryEntry[]
  future: EditorHistoryEntry[]
  maxHistorySize: number
}

interface ValidationState {
  isValid: boolean
  issues: ValidationIssue[]
}

type StepTypesStatus = "idle" | "loading" | "ready" | "error"

interface StepTypesState {
  stepRegistry: StepRegistry
  stepSchemas: Record<string, WorkflowStepSchema>
  stepTypesStatus: StepTypesStatus
  stepTypesError: string | null
}

// ─────────────────────────────────────────────────────────────────────────────
// Actions Types
// ─────────────────────────────────────────────────────────────────────────────

interface CanvasActions {
  // Node operations
  addNode: (params: {
    type: WorkflowStepType
    position: XYPosition
    data?: Partial<WorkflowNodeData>
  }) => string
  updateNode: (nodeId: string, updates: Partial<WorkflowNodeData>) => void
  updateNodePosition: (nodeId: string, position: XYPosition) => void
  deleteNodes: (nodeIds: string[]) => void
  duplicateNodes: (nodeIds: string[]) => void

  // Edge operations
  addEdge: (connection: Connection) => void
  deleteEdges: (edgeIds: string[]) => void

  // React Flow handlers
  onNodesChange: (changes: NodeChange<WorkflowNode>[]) => void
  onEdgesChange: (changes: EdgeChange<WorkflowEdge>[]) => void
  onConnect: (connection: Connection) => void

  // Workflow operations
  newWorkflow: () => void
  loadWorkflow: (workflow: ServerWorkflowDefinition) => void
  saveWorkflow: () => ServerWorkflowDefinition
  setWorkflowMeta: (name: string, description: string) => void
  clearCanvas: () => void
}

interface StepTypesActions {
  loadStepTypes: (force?: boolean) => Promise<void>
}

interface UIActions {
  setSelectedNodes: (nodeIds: string[]) => void
  setSelectedEdges: (edgeIds: string[]) => void
  selectNode: (nodeId: string, addToSelection?: boolean) => void
  deselectAll: () => void
  setActiveTool: (tool: EditorTool) => void
  setSidebarPanel: (panel: SidebarPanel) => void
  toggleMiniMap: () => void
  toggleGrid: () => void
  setZoom: (zoom: number) => void
  setPanPosition: (position: XYPosition) => void
  setDragging: (isDragging: boolean) => void
  setConnecting: (isConnecting: boolean) => void
  setLoading: (isLoading: boolean) => void
  setError: (error: string | null) => void
  fitView: () => void
}

interface ExecutionActions {
  startRun: (runId: string) => void
  stopRun: () => void
  pauseRun: () => void
  resumeRun: () => void
  updateStepState: (nodeId: string, state: Partial<StepExecutionState>) => void
  setStepStatus: (nodeId: string, status: StepExecutionStatus) => void
  setStepOutput: (nodeId: string, output: unknown) => void
  setStreamingOutput: (nodeId: string, chunk: string) => void
  clearStreamingOutput: (nodeId: string) => void
  setPendingApproval: (request: HumanApprovalRequest | null) => void
  respondToApproval: (response: HumanApprovalResponse) => void
  setRunError: (error: string | null) => void
  loadRunInvestigation: (runId: string) => Promise<void>
  clearRunInvestigation: () => void
  resetExecution: () => void
  setConnected: (connected: boolean) => void
}

interface HistoryActions {
  undo: () => void
  redo: () => void
  pushHistory: (description: string) => void
  canUndo: () => boolean
  canRedo: () => boolean
  clearHistory: () => void
}

interface ValidationActions {
  validate: () => ValidationState
  clearValidation: () => void
}

interface PersistenceActions {
  loadFromStorage: () => Promise<void>
  saveToStorage: () => Promise<void>
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined Store Type
// ─────────────────────────────────────────────────────────────────────────────

export type WorkflowEditorState = CanvasState &
  UIState &
  ExecutionState &
  HistoryState &
  ValidationState &
  StepTypesState &
  CanvasActions &
  UIActions &
  ExecutionActions &
  HistoryActions &
  ValidationActions &
  PersistenceActions &
  StepTypesActions

// ─────────────────────────────────────────────────────────────────────────────
// Initial State
// ─────────────────────────────────────────────────────────────────────────────

const initialCanvasState: CanvasState = {
  nodes: [],
  edges: [],
  workflowId: null,
  workflowVersion: null,
  savedCanvasKey: null,
  ownerScopeKey: null,
  viewRevision: 0,
  workflowName: "Untitled Workflow",
  workflowDescription: "",
  isDirty: false
}

const initialUIState: UIState = {
  selectedNodeIds: [],
  selectedEdgeIds: [],
  activeTool: "select",
  sidebarPanel: "palette",
  isMiniMapVisible: true,
  isGridVisible: true,
  zoom: 1,
  panPosition: { x: 0, y: 0 },
  isDragging: false,
  isConnecting: false,
  isLoading: false,
  error: null
}

const initialExecutionState: ExecutionState = {
  runId: null,
  runWorkflowId: null,
  runWorkflowVersion: null,
  pendingRunSubmission: null,
  status: "idle",
  startedAt: undefined,
  completedAt: undefined,
  nodeStates: {},
  currentNodeId: undefined,
  pendingApproval: null,
  error: undefined,
  isConnected: false,
  runInvestigation: null,
  runInvestigationLoading: false,
  runInvestigationError: null
}

const initialHistoryState: HistoryState = {
  past: [],
  future: [],
  maxHistorySize: 50
}

const initialValidationState: ValidationState = {
  isValid: true,
  issues: []
}

const initialStepTypesState: StepTypesState = {
  stepRegistry: BASE_STEP_REGISTRY,
  stepSchemas: {},
  stepTypesStatus: "idle",
  stepTypesError: null
}

const initialState = {
  ...initialCanvasState,
  ...initialUIState,
  ...initialExecutionState,
  ...initialHistoryState,
  ...initialValidationState,
  ...initialStepTypesState
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const generateId = () =>
  `node-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`

const createDefaultStepData = (type: WorkflowStepType): WorkflowNodeData => {
  const labels: Record<string, string> = {
    prompt: "LLM Prompt",
    rag_search: "RAG Search",
    media_ingest: "Media Ingest",
    branch: "Branch",
    map: "Map",
    wait_for_human: "Human Approval",
    wait_for_approval: "Approval",
    webhook: "Webhook",
    tts: "Text to Speech",
    stt_transcribe: "Transcribe",
    delay: "Delay",
    log: "Log",
    start: "Start",
    end: "End"
  }

  return {
    label: labels[type] || humanizeStepType(String(type)),
    stepType: type,
    config: {},
    isExpanded: false
  }
}

const createHistoryEntry = (
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  description: string
): EditorHistoryEntry => ({
  id: generateId(),
  timestamp: Date.now(),
  description,
  nodes: JSON.parse(JSON.stringify(nodes)),
  edges: JSON.parse(JSON.stringify(edges))
})

// Storage helpers
const loadFromStorage = async <T>(key: string, defaultValue: T): Promise<T> => {
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local) {
      const result = await chrome.storage.local.get(key)
      return result[key] ?? defaultValue
    }
    const stored = localStorage.getItem(key)
    return stored ? JSON.parse(stored) : defaultValue
  } catch {
    return defaultValue
  }
}

const saveToStorage = async <T>(key: string, value: T): Promise<void> => {
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local) {
      await chrome.storage.local.set({ [key]: value })
    } else {
      localStorage.setItem(key, JSON.stringify(value))
    }
  } catch (e) {
    console.error(`Failed to save ${key}:`, e)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useWorkflowEditorStore = createWithEqualityFn<WorkflowEditorState>()(
  (set, get) => ({
    ...initialState,

    // ─────────────────────────────────────────────────────────────────────────
    // Step Types (Server-Driven)
    // ─────────────────────────────────────────────────────────────────────────

    loadStepTypes: async (force = false) => {
      const status = get().stepTypesStatus
      if (status === "loading") return
      if (!force && status === "ready") return

      set({
        stepTypesStatus: "loading",
        stepTypesError: null
      })

      try {
        const steps = await getWorkflowStepTypes(force)
        const registry = buildStepRegistry(steps)
        const schemas: Record<string, WorkflowStepSchema> = {}
        for (const step of steps) {
          if (!step?.name) continue
          schemas[step.name] = step.schema || {}
        }
        set({
          stepRegistry: registry,
          stepSchemas: schemas,
          stepTypesStatus: "ready"
        })
      } catch (error) {
        console.warn("Failed to load workflow step types", error)
        set({
          stepRegistry: buildStepRegistry([]),
          stepSchemas: {},
          stepTypesStatus: "error",
          stepTypesError:
            error instanceof Error ? error.message : "Failed to load step types"
        })
      }
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Canvas Actions
    // ─────────────────────────────────────────────────────────────────────────

    addNode: ({ type, position, data }) => {
      const id = generateId()
      const defaultData = createDefaultStepData(type)
      const newNode: WorkflowNode = {
        id,
        type,
        position,
        data: { ...defaultData, ...data }
      }

      get().pushHistory(`Add ${type} node`)

      set((state) => ({
        nodes: [...state.nodes, newNode],
        isDirty: true,
        selectedNodeIds: [id],
        sidebarPanel: "config"
      }))

      return id
    },

    updateNode: (nodeId, updates) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, ...updates } }
            : node
        ),
        isDirty: true
      }))
    },

    updateNodePosition: (nodeId, position) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId ? { ...node, position } : node
        )
      }))
    },

    deleteNodes: (nodeIds) => {
      get().pushHistory(`Delete ${nodeIds.length} node(s)`)

      set((state) => ({
        nodes: state.nodes.filter((node) => !nodeIds.includes(node.id)),
        edges: state.edges.filter(
          (edge) =>
            !nodeIds.includes(edge.source) && !nodeIds.includes(edge.target)
        ),
        selectedNodeIds: state.selectedNodeIds.filter(
          (id) => !nodeIds.includes(id)
        ),
        isDirty: true
      }))
    },

    duplicateNodes: (nodeIds) => {
      const nodesToDuplicate = get().nodes.filter((n) => nodeIds.includes(n.id))
      if (nodesToDuplicate.length === 0) return

      get().pushHistory(`Duplicate ${nodeIds.length} node(s)`)

      const idMap = new Map<string, string>()
      const offset = { x: 50, y: 50 }

      const newNodes: WorkflowNode[] = nodesToDuplicate.map((node) => {
        const newId = generateId()
        idMap.set(node.id, newId)
        return {
          ...node,
          id: newId,
          position: {
            x: node.position.x + offset.x,
            y: node.position.y + offset.y
          },
          data: { ...node.data }
        }
      })

      // Duplicate edges between duplicated nodes
      const newEdges = get()
        .edges.filter(
          (edge) => nodeIds.includes(edge.source) && nodeIds.includes(edge.target)
        )
        .map((edge) => ({
          ...edge,
          id: `edge-${generateId()}`,
          source: idMap.get(edge.source) || edge.source,
          target: idMap.get(edge.target) || edge.target
        }))

      set((state) => ({
        nodes: [...state.nodes, ...newNodes],
        edges: [...state.edges, ...newEdges],
        selectedNodeIds: newNodes.map((n) => n.id),
        isDirty: true
      }))
    },

    addEdge: (connection) => {
      if (!connection.source || !connection.target) return

      get().pushHistory("Add connection")

      set((state) => ({
        edges: rfAddEdge(
          {
            ...connection,
            id: `edge-${generateId()}`,
            type: "smoothstep",
            animated: false
          },
          state.edges
        ) as WorkflowEdge[],
        isDirty: true
      }))
    },

    deleteEdges: (edgeIds) => {
      get().pushHistory(`Delete ${edgeIds.length} edge(s)`)

      set((state) => ({
        edges: state.edges.filter((edge) => !edgeIds.includes(edge.id)),
        selectedEdgeIds: state.selectedEdgeIds.filter(
          (id) => !edgeIds.includes(id)
        ),
        isDirty: true
      }))
    },

    onNodesChange: (changes: NodeChange[]) => {
      set((state) => ({
        nodes: applyNodeChanges(changes, state.nodes) as WorkflowNode[]
      }))
    },

    onEdgesChange: (changes) => {
      set((state) => ({
        edges: applyEdgeChanges(changes, state.edges)
      }))
    },

    onConnect: (connection) => {
      get().addEdge(connection)
    },

    newWorkflow: () => {
      const startNode: WorkflowNode = {
        id: generateId(),
        type: "start",
        position: { x: 100, y: 200 },
        data: createDefaultStepData("start")
      }

      const endNode: WorkflowNode = {
        id: generateId(),
        type: "end",
        position: { x: 500, y: 200 },
        data: createDefaultStepData("end")
      }

      set({
        ...initialState,
        ownerScopeKey: get().ownerScopeKey,
        viewRevision: get().viewRevision + 1,
        nodes: [startNode, endNode],
        edges: []
      })
    },

    loadWorkflow: (workflow) => {
      set({
        ...initialExecutionState,
        viewRevision: get().viewRevision + 1,
        nodes: workflow.nodes,
        edges: workflow.edges,
        workflowId: workflow.id || null,
        workflowVersion: workflow.version || null,
        savedCanvasKey: null,
        workflowName: workflow.name,
        workflowDescription: workflow.description || "",
        isDirty: false,
        selectedNodeIds: [],
        selectedEdgeIds: [],
        past: [],
        future: []
      })
    },

    saveWorkflow: () => {
      const { nodes, edges, workflowId, workflowName, workflowDescription } =
        get()

      const workflow: ServerWorkflowDefinition = {
        id: workflowId || undefined,
        name: workflowName,
        description: workflowDescription,
        nodes,
        edges,
        metadata: {
          updatedAt: Date.now()
        }
      }

      return workflow
    },

    setWorkflowMeta: (name, description) => {
      set({
        workflowName: name,
        workflowDescription: description,
        isDirty: true
      })
    },

    clearCanvas: () => {
      get().pushHistory("Clear canvas")
      set({
        ...initialExecutionState,
        viewRevision: get().viewRevision + 1,
        nodes: [],
        edges: [],
        selectedNodeIds: [],
        selectedEdgeIds: [],
        isDirty: true
      })
    },

    // ─────────────────────────────────────────────────────────────────────────
    // UI Actions
    // ─────────────────────────────────────────────────────────────────────────

    setSelectedNodes: (nodeIds) => set({ selectedNodeIds: nodeIds }),

    setSelectedEdges: (edgeIds) => set({ selectedEdgeIds: edgeIds }),

    selectNode: (nodeId, addToSelection = false) => {
      set((state) => ({
        selectedNodeIds: addToSelection
          ? [...state.selectedNodeIds, nodeId]
          : [nodeId],
        sidebarPanel: "config"
      }))
    },

    deselectAll: () => set({ selectedNodeIds: [], selectedEdgeIds: [] }),

    setActiveTool: (tool) => set({ activeTool: tool }),

    setSidebarPanel: (panel) => set({ sidebarPanel: panel }),

    toggleMiniMap: () =>
      set((state) => ({ isMiniMapVisible: !state.isMiniMapVisible })),

    toggleGrid: () => set((state) => ({ isGridVisible: !state.isGridVisible })),

    setZoom: (zoom) => set({ zoom }),

    setPanPosition: (position) => set({ panPosition: position }),

    setDragging: (isDragging) => set({ isDragging }),

    setConnecting: (isConnecting) => set({ isConnecting }),

    setLoading: (isLoading) => set({ isLoading }),

    setError: (error) => set({ error }),

    fitView: () => {
      // This will be called from the React Flow component
      // Just reset zoom to indicate fit view was requested
      set({ zoom: 1 })
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Execution Actions
    // ─────────────────────────────────────────────────────────────────────────

    startRun: (runId) => {
      set({
        runId,
        status: "running",
        startedAt: Date.now(),
        completedAt: undefined,
        nodeStates: {},
        currentNodeId: undefined,
        pendingApproval: null,
        error: undefined,
        runInvestigation: null,
        runInvestigationLoading: false,
        runInvestigationError: null
      })
    },

    stopRun: () => {
      set(() => ({
        status: "cancelled",
        completedAt: Date.now(),
        currentNodeId: undefined,
        runInvestigation: null,
        runInvestigationLoading: false,
        runInvestigationError: null
      }))
    },

    pauseRun: () => {
      set({ status: "paused" })
    },

    resumeRun: () => {
      set({ status: "running" })
    },

    updateStepState: (nodeId, state) => {
      set((s) => ({
        nodeStates: {
          ...s.nodeStates,
          [nodeId]: {
            ...s.nodeStates[nodeId],
            nodeId,
            ...state
          }
        }
      }))
    },

    setStepStatus: (nodeId, status) => {
      const currentNodeId = status === "running" ? nodeId : get().currentNodeId

      set((state) => ({
        nodeStates: {
          ...state.nodeStates,
          [nodeId]: {
            ...state.nodeStates[nodeId],
            nodeId,
            status,
            startedAt:
              status === "running"
                ? Date.now()
                : state.nodeStates[nodeId]?.startedAt,
            completedAt:
              status === "success" || status === "failed"
                ? Date.now()
                : undefined
          }
        },
        currentNodeId,
        status: status === "waiting_human" ? "waiting_human" : state.status
      }))
    },

    setStepOutput: (nodeId, output) => {
      set((state) => ({
        nodeStates: {
          ...state.nodeStates,
          [nodeId]: {
            ...state.nodeStates[nodeId],
            nodeId,
            output
          }
        }
      }))
    },

    setStreamingOutput: (nodeId, chunk) => {
      set((state) => ({
        nodeStates: {
          ...state.nodeStates,
          [nodeId]: {
            ...state.nodeStates[nodeId],
            nodeId,
            streamingOutput:
              (state.nodeStates[nodeId]?.streamingOutput || "") + chunk
          }
        }
      }))
    },

    clearStreamingOutput: (nodeId) => {
      set((state) => ({
        nodeStates: {
          ...state.nodeStates,
          [nodeId]: {
            ...state.nodeStates[nodeId],
            nodeId,
            streamingOutput: undefined
          }
        }
      }))
    },

    setPendingApproval: (request) => {
      set({
        pendingApproval: request,
        status: request ? "waiting_human" : get().status
      })
    },

    respondToApproval: (response) => {
      set({
        pendingApproval: null,
        status: "running"
      })
      // The actual response handling will be done by the execution service
    },

    setRunError: (error) => {
      set({
        error,
        status: error ? "failed" : get().status,
        completedAt: error ? Date.now() : undefined,
        runInvestigation: error ? null : get().runInvestigation,
        runInvestigationLoading: false,
        runInvestigationError: null
      })
    },

    loadRunInvestigation: async (runId) => {
      if (!runId) return
      const state = get()
      if (state.runInvestigationLoading) return
      if (
        state.runInvestigation?.run_id === runId &&
        !state.runInvestigationError
      ) {
        return
      }

      set({
        runInvestigationLoading: true,
        runInvestigationError: null
      })

      try {
        const investigation = await getWorkflowInvestigation(runId)
        if (get().runId !== runId) return
        set({
          runInvestigation: investigation,
          runInvestigationLoading: false,
          runInvestigationError: null
        })
      } catch (error) {
        if (get().runId !== runId) return
        set({
          runInvestigation: null,
          runInvestigationLoading: false,
          runInvestigationError:
            error instanceof Error
              ? error.message
              : "Could not load run diagnostics."
        })
      }
    },

    clearRunInvestigation: () => {
      set({
        runInvestigation: null,
        runInvestigationLoading: false,
        runInvestigationError: null
      })
    },

    resetExecution: () => {
      set(initialExecutionState)
    },

    setConnected: (connected) => {
      set({ isConnected: connected })
    },

    // ─────────────────────────────────────────────────────────────────────────
    // History Actions
    // ─────────────────────────────────────────────────────────────────────────

    undo: () => {
      const { past, nodes, edges, maxHistorySize } = get()
      if (past.length === 0) return

      const previous = past[past.length - 1]
      const current = createHistoryEntry(nodes, edges, "Current state")

      set({
        nodes: previous.nodes,
        edges: previous.edges,
        past: past.slice(0, -1),
        future: [current, ...get().future].slice(0, maxHistorySize),
        isDirty: true
      })
    },

    redo: () => {
      const { future, nodes, edges, maxHistorySize } = get()
      if (future.length === 0) return

      const next = future[0]
      const current = createHistoryEntry(nodes, edges, "Current state")

      set({
        nodes: next.nodes,
        edges: next.edges,
        future: future.slice(1),
        past: [...get().past, current].slice(-maxHistorySize),
        isDirty: true
      })
    },

    pushHistory: (description) => {
      const { nodes, edges, past, maxHistorySize } = get()
      const entry = createHistoryEntry(nodes, edges, description)

      set({
        past: [...past, entry].slice(-maxHistorySize),
        future: [] // Clear redo stack on new action
      })
    },

    canUndo: () => get().past.length > 0,

    canRedo: () => get().future.length > 0,

    clearHistory: () => {
      set({ past: [], future: [] })
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Validation Actions
    // ─────────────────────────────────────────────────────────────────────────

    validate: () => {
      const { nodes, edges } = get()
      const issues: ValidationIssue[] = []

      // Check for start node
      const startNodes = nodes.filter((n) => n.data.stepType === "start")
      if (startNodes.length === 0) {
        issues.push({
          id: "no-start",
          severity: "error",
          message: "Workflow must have a Start node"
        })
      } else if (startNodes.length > 1) {
        issues.push({
          id: "multiple-start",
          severity: "error",
          message: "Workflow can only have one Start node"
        })
      }

      // Check for end node
      const endNodes = nodes.filter((n) => n.data.stepType === "end")
      if (endNodes.length === 0) {
        issues.push({
          id: "no-end",
          severity: "error",
          message: "Workflow must have an End node"
        })
      }

      // Check for disconnected nodes
      nodes.forEach((node) => {
        if (node.data.stepType === "start") return

        const hasIncoming = edges.some((e) => e.target === node.id)
        if (!hasIncoming) {
          issues.push({
            id: `disconnected-${node.id}`,
            nodeId: node.id,
            severity: "warning",
            message: `Node "${node.data.label}" has no incoming connections`
          })
        }
      })

      // Check for nodes without outgoing (except end)
      nodes.forEach((node) => {
        if (node.data.stepType === "end") return

        const hasOutgoing = edges.some((e) => e.source === node.id)
        if (!hasOutgoing) {
          issues.push({
            id: `no-output-${node.id}`,
            nodeId: node.id,
            severity: "warning",
            message: `Node "${node.data.label}" has no outgoing connections`
          })
        }
      })

      const validation: ValidationState = {
        isValid: !issues.some((i) => i.severity === "error"),
        issues
      }

      set(validation)
      return validation
    },

    clearValidation: () => {
      set({ isValid: true, issues: [] })
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Persistence Actions
    // ─────────────────────────────────────────────────────────────────────────

    loadFromStorage: async () => {
      const saved = await loadFromStorage<{
        nodes: WorkflowNode[]
        edges: WorkflowEdge[]
        workflowName: string
        workflowDescription: string
      } | null>(EDITOR_STATE_KEY, null)

      if (saved) {
        set({
          nodes: saved.nodes,
          edges: saved.edges,
          workflowName: saved.workflowName,
          workflowDescription: saved.workflowDescription,
          isDirty: false
        })
      }
    },

    saveToStorage: async () => {
      const { nodes, edges, workflowName, workflowDescription } = get()
      await saveToStorage(EDITOR_STATE_KEY, {
        nodes,
        edges,
        workflowName,
        workflowDescription
      })
    }
  })
)

// Expose for debugging
if (typeof window !== "undefined") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useWorkflowEditorStore = useWorkflowEditorStore
}
