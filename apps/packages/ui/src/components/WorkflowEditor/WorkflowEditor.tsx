/**
 * WorkflowEditor Component
 *
 * Main workflow editor component that combines:
 * - Toolbar with workflow actions
 * - Canvas for visual editing
 * - Sidebar with palette, config, and execution panels
 */

import { useEffect, useCallback, useState, lazy, Suspense } from "react"
import {
  Button,
  Input,
  Tooltip,
  Dropdown,
  Segmented,
  Badge,
  Drawer,
  message
} from "antd"
import {
  Save,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Grid3X3,
  Map,
  FileDown,
  FileUp,
  Settings,
  Palette,
  Play,
  MoreVertical,
  AlertCircle,
  Plus,
  Trash2,
  PanelLeft
} from "lucide-react"
import type { SidebarPanel } from "@/types/workflow-editor"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { useDesktop } from "@/hooks/useMediaQuery"
import { WorkflowCanvas } from "./WorkflowCanvas"
import { NodePalette } from "./NodePalette"

const LazyNodeConfigPanel = lazy(() =>
  import("./NodeConfigPanel").then((module) => ({
    default: module.NodeConfigPanel
  }))
)

const LazyExecutionPanel = lazy(() =>
  import("./ExecutionPanel").then((module) => ({
    default: module.ExecutionPanel
  }))
)

interface WorkflowEditorProps {
  className?: string
}

export const WorkflowEditor = ({ className = "" }: WorkflowEditorProps) => {
  const isDesktop = useDesktop()

  // Store state
  const workflowName = useWorkflowEditorStore((s) => s.workflowName)
  const isDirty = useWorkflowEditorStore((s) => s.isDirty)
  const nodes = useWorkflowEditorStore((s) => s.nodes)
  const edges = useWorkflowEditorStore((s) => s.edges)
  const isMiniMapVisible = useWorkflowEditorStore((s) => s.isMiniMapVisible)
  const isGridVisible = useWorkflowEditorStore((s) => s.isGridVisible)
  const sidebarPanel = useWorkflowEditorStore((s) => s.sidebarPanel)
  const isValid = useWorkflowEditorStore((s) => s.isValid)
  const issues = useWorkflowEditorStore((s) => s.issues)
  const status = useWorkflowEditorStore((s) => s.status)
  const stepTypesStatus = useWorkflowEditorStore((s) => s.stepTypesStatus)
  const stepTypesError = useWorkflowEditorStore((s) => s.stepTypesError)

  // Store actions
  const setWorkflowMeta = useWorkflowEditorStore((s) => s.setWorkflowMeta)
  const newWorkflow = useWorkflowEditorStore((s) => s.newWorkflow)
  const saveWorkflow = useWorkflowEditorStore((s) => s.saveWorkflow)
  const toggleMiniMap = useWorkflowEditorStore((s) => s.toggleMiniMap)
  const toggleGrid = useWorkflowEditorStore((s) => s.toggleGrid)
  const setSidebarPanel = useWorkflowEditorStore((s) => s.setSidebarPanel)
  const undo = useWorkflowEditorStore((s) => s.undo)
  const redo = useWorkflowEditorStore((s) => s.redo)
  const canUndo = useWorkflowEditorStore((s) => s.canUndo)
  const canRedo = useWorkflowEditorStore((s) => s.canRedo)
  const validate = useWorkflowEditorStore((s) => s.validate)
  const clearCanvas = useWorkflowEditorStore((s) => s.clearCanvas)
  const loadStepTypes = useWorkflowEditorStore((s) => s.loadStepTypes)

  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(workflowName)
  const [isMobilePanelsOpen, setIsMobilePanelsOpen] = useState(false)

  useEffect(() => {
    if (isDesktop) {
      setIsMobilePanelsOpen(false)
    }
  }, [isDesktop])

  // Initialize with a new workflow on mount
  useEffect(() => {
    const nodes = useWorkflowEditorStore.getState().nodes
    if (nodes.length === 0) {
      newWorkflow()
    }
  }, [newWorkflow])

  // Load server-driven step types once
  useEffect(() => {
    if (stepTypesStatus === "idle") {
      void loadStepTypes()
    }
  }, [stepTypesStatus, loadStepTypes])

  // Validate on graph/config changes with a short debounce for typing comfort
  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      validate()
    }, 120)
    return () => window.clearTimeout(timeoutId)
  }, [nodes, edges, validate])

  const handleSave = useCallback(() => {
    const workflow = saveWorkflow()
    console.log("Workflow saved:", workflow)
    message.success("Workflow saved")
    // In a real implementation, this would call the API
  }, [saveWorkflow])

  const handleExport = useCallback(() => {
    const workflow = saveWorkflow()
    const blob = new Blob([JSON.stringify(workflow, null, 2)], {
      type: "application/json"
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${workflow.name.toLowerCase().replace(/\s+/g, "-")}.json`
    a.click()
    URL.revokeObjectURL(url)
    message.success("Workflow exported")
  }, [saveWorkflow])

  const handleImport = useCallback(() => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = ".json"
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      try {
        const text = await file.text()
        const workflow = JSON.parse(text)
        useWorkflowEditorStore.getState().loadWorkflow(workflow)
        message.success("Workflow imported")
      } catch (error) {
        message.error("Failed to import workflow")
      }
    }
    input.click()
  }, [])

  const handleOpenRunPanel = useCallback(() => {
    setSidebarPanel("execution")
    if (!isDesktop) {
      setIsMobilePanelsOpen(true)
    }
  }, [isDesktop, setSidebarPanel])

  const handleNameEdit = useCallback(() => {
    if (isEditing) {
      setWorkflowMeta(editName, useWorkflowEditorStore.getState().workflowDescription)
    }
    setIsEditing(!isEditing)
  }, [isEditing, editName, setWorkflowMeta])

  const sidebarPanelOptions = [
    { value: "palette", icon: <Palette className="w-4 h-4" />, label: "Nodes" },
    { value: "config", icon: <Settings className="w-4 h-4" />, label: "Config" },
    { value: "execution", icon: <Play className="w-4 h-4" />, label: "Run" }
  ]

  const moreMenuItems = [
    {
      key: "new",
      icon: <Plus className="w-4 h-4" />,
      label: "New Workflow",
      onClick: () => {
        if (isDirty) {
          if (confirm("You have unsaved changes. Create new workflow?")) {
            newWorkflow()
          }
        } else {
          newWorkflow()
        }
      }
    },
    {
      key: "import",
      icon: <FileUp className="w-4 h-4" />,
      label: "Import",
      onClick: handleImport
    },
    {
      key: "export",
      icon: <FileDown className="w-4 h-4" />,
      label: "Export",
      onClick: handleExport
    },
    { type: "divider" as const },
    {
      key: "clear",
      icon: <Trash2 className="w-4 h-4" />,
      label: "Clear Canvas",
      danger: true,
      onClick: () => {
        if (confirm("Clear all nodes from the canvas?")) {
          clearCanvas()
        }
      }
    }
  ]

  const errorCount = issues.filter((i) => i.severity === "error").length
  const warningCount = issues.filter((i) => i.severity === "warning").length
  const stepTypesStatusLabel =
    stepTypesStatus === "loading"
      ? "Step types loading"
      : stepTypesStatus === "error"
        ? "Step types unavailable"
        : stepTypesStatus === "ready"
          ? "Step types ready"
          : "Step types pending"
  const validationSummaryLabel =
    errorCount > 0 && warningCount > 0
      ? `${errorCount} error${errorCount === 1 ? "" : "s"}, ${warningCount} warning${
          warningCount === 1 ? "" : "s"
        }`
      : errorCount > 0
        ? `${errorCount} error${errorCount === 1 ? "" : "s"}`
        : warningCount > 0
          ? `${warningCount} warning${warningCount === 1 ? "" : "s"}`
          : "Validation clear"
  const validationIssuesAriaLabel =
    errorCount > 0 && warningCount > 0
      ? `Validation issues: ${errorCount} errors, ${warningCount} warnings`
      : errorCount > 0
        ? `Validation issues: ${errorCount} errors`
        : `Validation issues: ${warningCount} warnings`

  const renderSidebarPanel = () => {
    if (sidebarPanel === "config") {
      return (
        <Suspense fallback={null}>
          <LazyNodeConfigPanel className="h-full" />
        </Suspense>
      )
    }

    if (sidebarPanel === "execution") {
      return (
        <Suspense fallback={null}>
          <LazyExecutionPanel className="h-full" />
        </Suspense>
      )
    }

    return <NodePalette className="h-full" />
  }

  const sidebarContent = (
    <>
      <div className="flex items-center p-2 border-b border-border">
        <Segmented
          size="small"
          value={sidebarPanel || "palette"}
          onChange={(value) => setSidebarPanel(value as SidebarPanel)}
          options={sidebarPanelOptions.map((opt) => ({
            value: opt.value,
            label: (
              <Tooltip title={opt.label}>
                <span className="inline-flex items-center justify-center">
                  <span aria-hidden="true">{opt.icon}</span>
                  <span className="sr-only">{opt.label}</span>
                </span>
              </Tooltip>
            ),
            title: opt.label
          }))}
          block
        />
      </div>

      <div className="flex-1 overflow-hidden">
        {renderSidebarPanel()}
      </div>
    </>
  )

  return (
    <div className={`flex flex-col h-full bg-bg ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-2 bg-surface border-b border-border">
        {/* Workflow name */}
        <div className="flex items-center gap-2 min-w-[200px]">
          {isEditing ? (
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onBlur={handleNameEdit}
              onPressEnter={handleNameEdit}
              size="small"
              autoFocus
              className="max-w-[180px]"
            />
          ) : (
            <button
              onClick={() => {
                setEditName(workflowName)
                setIsEditing(true)
              }}
              className="text-sm font-medium text-text hover:text-primary truncate max-w-[180px]"
            >
              {workflowName}
            </button>
          )}
          {isDirty && (
            <span className="text-xs text-text-subtle">*</span>
          )}
        </div>

        <div className="h-4 w-px bg-border" />

        {/* Undo/Redo */}
        <div className="flex items-center gap-1">
          <Tooltip title="Undo (Cmd+Z)">
            <Button
              type="text"
              size="small"
              aria-label="Undo"
              icon={<Undo2 className="w-4 h-4" />}
              disabled={!canUndo()}
              onClick={undo}
            />
          </Tooltip>
          <Tooltip title="Redo (Cmd+Shift+Z)">
            <Button
              type="text"
              size="small"
              aria-label="Redo"
              icon={<Redo2 className="w-4 h-4" />}
              disabled={!canRedo()}
              onClick={redo}
            />
          </Tooltip>
        </div>

        <div className="h-4 w-px bg-border" />

        {/* View controls */}
        <div className="flex items-center gap-1">
          {!isDesktop && (
            <Tooltip title="Open workflow panels">
              <Button
                type="text"
                size="small"
                aria-label="Open workflow panels"
                icon={<PanelLeft className="w-4 h-4" />}
                onClick={() => setIsMobilePanelsOpen(true)}
              />
            </Tooltip>
          )}
          <Tooltip title="Toggle Grid">
            <Button
              type={isGridVisible ? "primary" : "text"}
              size="small"
              aria-label="Toggle Grid"
              icon={<Grid3X3 className="w-4 h-4" />}
              onClick={toggleGrid}
            />
          </Tooltip>
          <Tooltip title="Toggle Minimap">
            <Button
              type={isMiniMapVisible ? "primary" : "text"}
              size="small"
              aria-label="Toggle Minimap"
              icon={<Map className="w-4 h-4" />}
              onClick={toggleMiniMap}
            />
          </Tooltip>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Validation status */}
        {(errorCount > 0 || warningCount > 0) && (
          <Tooltip
            title={
              <div className="space-y-1">
                {issues.map((issue) => (
                  <div key={issue.id} className="flex items-start gap-1">
                    <AlertCircle className="w-3 h-3 shrink-0 mt-0.5" />
                    <span>{issue.message}</span>
                  </div>
                ))}
              </div>
            }
          >
            <Badge
              count={errorCount || warningCount}
              color={errorCount > 0 ? "red" : "orange"}
              size="small"
            >
              <Button
                type="text"
                size="small"
                aria-label={validationIssuesAriaLabel}
                icon={
                  <AlertCircle
                    className={`w-4 h-4 ${
                      errorCount > 0 ? "text-danger" : "text-warn"
                    }`}
                  />
                }
              />
            </Badge>
          </Tooltip>
        )}

        {/* Save button */}
        <Tooltip title="Save (Cmd+S)">
          <Button
            type="primary"
            size="small"
            icon={<Save className="w-4 h-4" />}
            onClick={handleSave}
            disabled={!isDirty}
          >
            Save
          </Button>
        </Tooltip>

        {/* More menu */}
        <Dropdown
          menu={{ items: moreMenuItems }}
          trigger={["click"]}
          placement="bottomRight"
        >
          <Tooltip title="More actions">
            <Button
              type="text"
              size="small"
              aria-label="More actions"
              icon={<MoreVertical className="w-4 h-4" />}
            />
          </Tooltip>
        </Dropdown>
      </div>

      <section
        aria-label="Workflow editor status summary"
        className="flex flex-wrap items-center gap-2 border-b border-border bg-surface2 px-3 py-2 text-xs text-text-muted"
        data-testid="workflow-editor-status-summary"
      >
        <span className="inline-flex items-center rounded-full border border-border bg-surface px-2 py-0.5 font-medium text-text">
          {stepTypesStatusLabel}
        </span>
        {stepTypesError ? (
          <span className="max-w-full truncate text-state-error">{stepTypesError}</span>
        ) : null}
        <span className="inline-flex items-center rounded-full border border-border bg-surface px-2 py-0.5 font-medium text-text">
          {validationSummaryLabel}
        </span>
        <span className="inline-flex items-center rounded-full border border-border bg-surface px-2 py-0.5 font-medium text-text">
          {isDirty ? "Unsaved changes" : "Saved"}
        </span>
        <span className="inline-flex items-center rounded-full border border-border bg-surface px-2 py-0.5 font-medium text-text">
          Run {status}
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-1">
          <Button
            size="small"
            aria-label="Import workflow"
            icon={<FileUp className="w-4 h-4" />}
            onClick={handleImport}
          >
            Import
          </Button>
          <Button
            size="small"
            aria-label="Export workflow"
            icon={<FileDown className="w-4 h-4" />}
            onClick={handleExport}
          >
            Export
          </Button>
          <Button
            size="small"
            aria-label="Open run panel"
            icon={<Play className="w-4 h-4" />}
            onClick={handleOpenRunPanel}
          >
            Run
          </Button>
        </div>
      </section>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar (desktop only) */}
        {isDesktop && (
          <div className="w-72 flex flex-col bg-surface border-r border-border">
            {sidebarContent}
          </div>
        )}

        {/* Canvas */}
        <div className="flex-1 overflow-hidden">
          <WorkflowCanvas />
        </div>
      </div>

      {!isDesktop && (
        <Drawer
          title="Workflow panels"
          placement="left"
          open={isMobilePanelsOpen}
          onClose={() => setIsMobilePanelsOpen(false)}
          styles={{
            wrapper: { width: 320 },
            body: { padding: 0, display: "flex", flexDirection: "column" }
          }}
        >
          {sidebarContent}
        </Drawer>
      )}

      {/* Status bar */}
      <div className="flex items-center justify-between px-3 py-1 bg-surface border-t border-border text-xs text-text-subtle">
        <div className="flex items-center gap-3">
          <span>
            {useWorkflowEditorStore.getState().nodes.length} nodes
          </span>
          <span>
            {useWorkflowEditorStore.getState().edges.length} connections
          </span>
        </div>
        <div className="flex items-center gap-2">
          {status !== "idle" && (
            <span className="flex items-center gap-1">
              <span
                className={`w-2 h-2 rounded-full ${
                  status === "running"
                    ? "bg-primary animate-pulse"
                    : status === "completed"
                    ? "bg-success"
                    : status === "failed"
                    ? "bg-danger"
                    : "bg-warn"
                }`}
              />
              {status}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export default WorkflowEditor
