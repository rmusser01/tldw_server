/**
 * WorkflowCanvas Component
 *
 * The main React Flow canvas for the workflow editor.
 * Handles node rendering, connections, and canvas interactions.
 */

import { useCallback, useRef, useMemo, useEffect } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  type Connection,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type OnSelectionChangeFunc,
  type ReactFlowInstance,
  BackgroundVariant,
  SelectionMode,
  type Node
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { message } from "antd"

import { getComputedTokens } from "@/themes/runtime-tokens"
import type { WorkflowStepType, WorkflowNode, WorkflowNodeData } from "@/types/workflow-editor"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { buildWorkflowNodeTypes } from "./nodes/WorkflowNode"
import { getStepMetadata } from "./step-registry"
import { validateWorkflowConnection } from "./connection-validation"
import { isEditableTarget } from "@/utils/editable-target"

interface WorkflowCanvasProps {
  className?: string
}

const WorkflowCanvasInner = ({ className = "" }: WorkflowCanvasProps) => {
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null)
  const { screenToFlowPosition } = useReactFlow()

  // Store state
  const nodes = useWorkflowEditorStore((s) => s.nodes)
  const edges = useWorkflowEditorStore((s) => s.edges)
  const isMiniMapVisible = useWorkflowEditorStore((s) => s.isMiniMapVisible)
  const isGridVisible = useWorkflowEditorStore((s) => s.isGridVisible)
  const stepRegistry = useWorkflowEditorStore((s) => s.stepRegistry)

  // Store actions
  const onNodesChange = useWorkflowEditorStore((s) => s.onNodesChange)
  const onEdgesChange = useWorkflowEditorStore((s) => s.onEdgesChange)
  const onConnect = useWorkflowEditorStore((s) => s.onConnect)
  const addNode = useWorkflowEditorStore((s) => s.addNode)
  const setSelectedNodes = useWorkflowEditorStore((s) => s.setSelectedNodes)
  const setSelectedEdges = useWorkflowEditorStore((s) => s.setSelectedEdges)
  const setZoom = useWorkflowEditorStore((s) => s.setZoom)
  const setPanPosition = useWorkflowEditorStore((s) => s.setPanPosition)

  const tokens = useMemo(() => getComputedTokens(), [])

  const nodeTypes = useMemo(() => {
    const registryTypes = Object.keys(stepRegistry || {})
    const nodeTypesFromNodes = nodes.map((node) => String(node.type || ""))
    const allTypes = Array.from(new Set([...registryTypes, ...nodeTypesFromNodes]))
    return buildWorkflowNodeTypes(allTypes)
  }, [stepRegistry, nodes])

  // Handle selection changes
  const handleSelectionChange: OnSelectionChangeFunc = useCallback(
    ({ nodes, edges }) => {
      setSelectedNodes(nodes.map((n) => n.id))
      setSelectedEdges(edges.map((e) => e.id))
    },
    [setSelectedNodes, setSelectedEdges]
  )

  // Handle drop from node palette
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = "copy"
  }, [])

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const stepType = event.dataTransfer.getData(
        "application/workflow-step"
      ) as WorkflowStepType

      if (!stepType) return

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY
      })

      addNode({ type: stepType, position })
    },
    [screenToFlowPosition, addNode]
  )

  const isValidConnection = useCallback(
    (connection: Connection) =>
      validateWorkflowConnection(connection, nodes, stepRegistry).valid,
    [nodes, stepRegistry]
  )

  const handleConnect: OnConnect = useCallback(
    (connection) => {
      const validation = validateWorkflowConnection(connection, nodes, stepRegistry)
      if (!validation.valid) {
        message.warning(
          validation.reason || "Invalid connection. Check source and target ports."
        )
        return
      }
      onConnect(connection)
    },
    [nodes, stepRegistry, onConnect]
  )

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) {
        return
      }

      // Delete selected nodes/edges
      if (e.key === "Delete" || e.key === "Backspace") {
        const selectedNodes = useWorkflowEditorStore.getState().selectedNodeIds
        const selectedEdges = useWorkflowEditorStore.getState().selectedEdgeIds

        if (selectedNodes.length > 0) {
          useWorkflowEditorStore.getState().deleteNodes(selectedNodes)
        }
        if (selectedEdges.length > 0) {
          useWorkflowEditorStore.getState().deleteEdges(selectedEdges)
        }
      }

      // Undo
      if ((e.metaKey || e.ctrlKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault()
        useWorkflowEditorStore.getState().undo()
      }

      // Redo
      if ((e.metaKey || e.ctrlKey) && e.key === "z" && e.shiftKey) {
        e.preventDefault()
        useWorkflowEditorStore.getState().redo()
      }

      // Duplicate
      if ((e.metaKey || e.ctrlKey) && e.key === "d") {
        e.preventDefault()
        const selectedNodes = useWorkflowEditorStore.getState().selectedNodeIds
        if (selectedNodes.length > 0) {
          useWorkflowEditorStore.getState().duplicateNodes(selectedNodes)
        }
      }

      // Select all
      if ((e.metaKey || e.ctrlKey) && e.key === "a") {
        e.preventDefault()
        const nodes = useWorkflowEditorStore.getState().nodes
        setSelectedNodes(nodes.map((n) => n.id))
      }

      // Escape to deselect
      if (e.key === "Escape") {
        useWorkflowEditorStore.getState().deselectAll()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [setSelectedNodes])

  // Handle viewport changes
  const handleMoveEnd = useCallback(
    (event: any, viewport: { x: number; y: number; zoom: number }) => {
      setZoom(viewport.zoom)
      setPanPosition({ x: viewport.x, y: viewport.y })
    },
    [setZoom, setPanPosition]
  )

  // Node colors for minimap
  const nodeColor = useCallback((node: Node) => {
    const data = node.data as WorkflowNodeData
    const meta = getStepMetadata(data.stepType, stepRegistry)
    // theme-exempt: workflow category colors
    const categoryColors: Record<string, string> = {
      ai: "#3b82f6",
      search: "#3b82f6",
      media: "#6366f1",
      text: "#06b6d4",
      research: "#8b5cf6",
      audio: "#14b8a6",
      video: "#10b981",
      control: "#6366f1",
      io: "#22c55e",
      utility: "#6b7280"
    }
    return categoryColors[meta?.category || "utility"] || "#6b7280"
  }, [stepRegistry])

  return (
    <div className={`h-full w-full ${className}`}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange as OnNodesChange<WorkflowNode>}
        onEdgesChange={onEdgesChange as OnEdgesChange}
        onConnect={handleConnect}
        isValidConnection={isValidConnection}
        onSelectionChange={handleSelectionChange}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onMoveEnd={handleMoveEnd}
        onInit={(instance) => {
          reactFlowInstance.current = instance
        }}
        nodeTypes={nodeTypes}
        selectionMode={SelectionMode.Partial}
        selectNodesOnDrag={false}
        panOnScroll
        zoomOnScroll
        panOnDrag
        fitView
        fitViewOptions={{
          padding: 0.2,
          maxZoom: 1.5
        }}
        defaultEdgeOptions={{
          type: "smoothstep",
          animated: false,
          style: {
            strokeWidth: 2,
            stroke: tokens.border
          }
        }}
        connectionLineStyle={{
          strokeWidth: 2,
          stroke: tokens.primary
        }}
        proOptions={{
          hideAttribution: true
        }}
        className="bg-bg"
      >
        {isGridVisible && (
          <Background
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
            color={tokens.muted}
            className="opacity-30"
          />
        )}

        <Controls
          showZoom
          showFitView
          showInteractive={false}
          className="!bg-surface !border-border !shadow-lg"
        />

        {isMiniMapVisible && (
          <MiniMap
            nodeColor={nodeColor}
            maskColor="rgba(0, 0, 0, 0.1)"
            className="!bg-surface !border-border"
          />
        )}
      </ReactFlow>
    </div>
  )
}

// Wrap with ReactFlowProvider
export const WorkflowCanvas = (props: WorkflowCanvasProps) => (
  <ReactFlowProvider>
    <WorkflowCanvasInner {...props} />
  </ReactFlowProvider>
)

export default WorkflowCanvas
