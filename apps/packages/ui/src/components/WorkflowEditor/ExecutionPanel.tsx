/**
 * ExecutionPanel Component
 *
 * Displays workflow execution status, step progress,
 * and handles human-in-the-loop approvals.
 */

import { useEffect, useMemo } from "react"
import {
  Button,
  Progress,
  Timeline,
  Empty,
  Modal,
  Input,
  Spin,
  Tag
} from "antd"
import {
  Play,
  Square,
  Pause,
  RotateCcw,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  AlertCircle,
  Hand,
  ChevronRight
} from "lucide-react"
import { WorkflowRunInspector } from "@/components/Common/Workflow"
import { Alert } from "@/components/ui/primitives"
import type { StepExecutionStatus } from "@/types/workflow-editor"
import { useWorkflowEditorStore } from "@/store/workflow-editor"

const { TextArea } = Input

// Status configuration
const STATUS_CONFIG: Record<
  StepExecutionStatus,
  { color: string; icon: React.ReactNode; label: string }
> = {
  idle: {
    color: "gray",
    icon: <Clock className="w-4 h-4" />,
    label: "Idle"
  },
  queued: {
    color: "blue",
    icon: <Clock className="w-4 h-4" />,
    label: "Queued"
  },
  running: {
    color: "processing",
    icon: <Loader2 className="w-4 h-4 animate-spin" />,
    label: "Running"
  },
  success: {
    color: "success",
    icon: <CheckCircle2 className="w-4 h-4" />,
    label: "Success"
  },
  failed: {
    color: "error",
    icon: <XCircle className="w-4 h-4" />,
    label: "Failed"
  },
  skipped: {
    color: "default",
    icon: <ChevronRight className="w-4 h-4" />,
    label: "Skipped"
  },
  waiting_human: {
    color: "warning",
    icon: <Hand className="w-4 h-4" />,
    label: "Waiting"
  },
  waiting_approval: {
    color: "warning",
    icon: <Hand className="w-4 h-4" />,
    label: "Waiting Approval"
  },
  cancelled: {
    color: "default",
    icon: <XCircle className="w-4 h-4" />,
    label: "Cancelled"
  }
}

interface ExecutionPanelProps {
  className?: string
}

export const ExecutionPanel = ({ className = "" }: ExecutionPanelProps) => {
  // Store state
  const nodes = useWorkflowEditorStore((s) => s.nodes)
  const runId = useWorkflowEditorStore((s) => s.runId)
  const status = useWorkflowEditorStore((s) => s.status)
  const nodeStates = useWorkflowEditorStore((s) => s.nodeStates)
  const pendingApproval = useWorkflowEditorStore((s) => s.pendingApproval)
  const error = useWorkflowEditorStore((s) => s.error)
  const startedAt = useWorkflowEditorStore((s) => s.startedAt)
  const completedAt = useWorkflowEditorStore((s) => s.completedAt)
  const runInvestigation = useWorkflowEditorStore((s) => s.runInvestigation)
  const runInvestigationLoading = useWorkflowEditorStore((s) => s.runInvestigationLoading)
  const runInvestigationError = useWorkflowEditorStore((s) => s.runInvestigationError)

  // Store actions
  const startRun = useWorkflowEditorStore((s) => s.startRun)
  const stopRun = useWorkflowEditorStore((s) => s.stopRun)
  const pauseRun = useWorkflowEditorStore((s) => s.pauseRun)
  const resumeRun = useWorkflowEditorStore((s) => s.resumeRun)
  const resetExecution = useWorkflowEditorStore((s) => s.resetExecution)
  const respondToApproval = useWorkflowEditorStore((s) => s.respondToApproval)
  const validate = useWorkflowEditorStore((s) => s.validate)
  const loadRunInvestigation = useWorkflowEditorStore((s) => s.loadRunInvestigation)

  // Calculate progress
  const progress = useMemo(() => {
    const total = nodes.length
    if (total === 0) return 0

    const completed = Object.values(nodeStates).filter(
      (s) => s.status === "success" || s.status === "skipped"
    ).length

    return Math.round((completed / total) * 100)
  }, [nodes, nodeStates])

  // Calculate elapsed time
  const elapsed = useMemo(() => {
    if (!startedAt) return null
    const end = completedAt || Date.now()
    const seconds = Math.floor((end - startedAt) / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`
  }, [startedAt, completedAt])

  // Build timeline items from node states
  const timelineItems = useMemo(() => {
    return nodes
      .filter((node) => nodeStates[node.id])
      .map((node) => {
        const state = nodeStates[node.id]
        const config = STATUS_CONFIG[state.status]

        return {
          key: node.id,
          color: config.color,
          dot: config.icon,
          children: (
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="font-medium text-sm">
                  {node.data.label}
                </div>
                {state.durationMs && (
                  <div className="text-xs text-text-subtle">
                    {(state.durationMs / 1000).toFixed(2)}s
                  </div>
                )}
                {state.error && (
                  <div className="text-xs text-danger mt-1">
                    {state.error}
                  </div>
                )}
              </div>
              <Tag color={config.color}>{config.label}</Tag>
            </div>
          )
        }
      })
  }, [nodes, nodeStates])

  const handleStartRun = () => {
    const validation = validate()
    if (!validation.isValid) {
      Modal.error({
        title: "Validation Failed",
        content: (
          <div className="space-y-2">
            {validation.issues.map((issue) => (
              <div key={issue.id} className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
                <span>{issue.message}</span>
              </div>
            ))}
          </div>
        )
      })
      return
    }

    // Generate run ID and start
    const newRunId = `run-${Date.now()}`
    startRun(newRunId)

    // In a real implementation, this would call the workflow API
    console.log("Starting workflow run:", newRunId)
  }

  const handleApprove = () => {
    if (!pendingApproval) return
    respondToApproval({
      requestId: pendingApproval.id,
      action: "approve",
      respondedAt: Date.now()
    })
  }

  const handleReject = () => {
    if (!pendingApproval) return
    respondToApproval({
      requestId: pendingApproval.id,
      action: "reject",
      reason: "Rejected by user",
      respondedAt: Date.now()
    })
  }

  const isRunning = status === "running"
  const isPaused = status === "paused"
  const isWaiting = status === "waiting_human" || status === "waiting_approval"
  const isIdle = status === "idle"
  const isCompleted = status === "completed" || status === "failed" || status === "cancelled"

  useEffect(() => {
    if (status === "failed" && runId) {
      void loadRunInvestigation(runId)
    }
  }, [loadRunInvestigation, runId, status])

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-text-muted flex items-center gap-2">
            <Play className="w-4 h-4" />
            Execution
          </h3>
          {elapsed && (
            <span className="text-xs text-text-subtle">{elapsed}</span>
          )}
        </div>

        {/* Control buttons */}
        <div className="flex items-center gap-2">
          {isIdle && (
            <Button
              type="primary"
              size="small"
              icon={<Play className="w-4 h-4" />}
              onClick={handleStartRun}
              className="flex-1"
            >
              Run
            </Button>
          )}

          {isRunning && (
            <>
              <Button
                size="small"
                icon={<Pause className="w-4 h-4" />}
                onClick={pauseRun}
                className="flex-1"
              >
                Pause
              </Button>
              <Button
                danger
                size="small"
                icon={<Square className="w-4 h-4" />}
                onClick={stopRun}
              >
                Stop
              </Button>
            </>
          )}

          {isPaused && (
            <>
              <Button
                type="primary"
                size="small"
                icon={<Play className="w-4 h-4" />}
                onClick={resumeRun}
                className="flex-1"
              >
                Resume
              </Button>
              <Button
                danger
                size="small"
                icon={<Square className="w-4 h-4" />}
                onClick={stopRun}
              >
                Stop
              </Button>
            </>
          )}

          {isCompleted && (
            <Button
              size="small"
              icon={<RotateCcw className="w-4 h-4" />}
              onClick={resetExecution}
              className="flex-1"
            >
              Reset
            </Button>
          )}
        </div>

        {/* Progress bar */}
        {(isRunning || isPaused || isWaiting) && (
          <Progress
            percent={progress}
            size="small"
            status={isWaiting ? "active" : isRunning ? "active" : "normal"}
            className="mt-3"
          />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {/* Error display */}
        {error && (
          <Alert
            variant="error"
            title="Execution Error"
            className="mb-3"
          >
            {error}
          </Alert>
        )}

        {status === "failed" && runId && (
          <div className="mb-4">
            {runInvestigationLoading ? (
              <div className="rounded-lg border border-border p-3">
                <Spin size="small" />
                <span className="ml-2 text-xs text-text-subtle">
                  Loading run diagnostics...
                </span>
              </div>
            ) : runInvestigationError ? (
              <Alert
                variant="warning"
                title="Diagnostics unavailable"
                action={{
                  label: "Retry diagnostics",
                  onClick: () => void loadRunInvestigation(runId)
                }}
              >
                {runInvestigationError}
              </Alert>
            ) : runInvestigation ? (
              <WorkflowRunInspector investigation={runInvestigation} />
            ) : null}
          </div>
        )}

        {/* Pending approval */}
        {pendingApproval && (
          <div className="mb-4 p-3 bg-warn/10 rounded-lg border border-warn/30">
            <div className="flex items-start gap-2 mb-2">
              <Hand className="w-5 h-5 text-warn shrink-0" />
              <div>
                <div className="font-medium text-sm">
                  Approval Required
                </div>
                <p className="text-xs text-text-muted mt-1">
                  {pendingApproval.promptMessage}
                </p>
              </div>
            </div>

            {/* Data preview */}
            <div className="mt-2 p-2 bg-surface rounded text-xs font-mono max-h-32 overflow-y-auto">
              {JSON.stringify(pendingApproval.dataToReview, null, 2)}
            </div>

            <div className="flex items-center gap-2 mt-3">
              <Button
                type="primary"
                size="small"
                icon={<CheckCircle2 className="w-4 h-4" />}
                onClick={handleApprove}
                className="flex-1"
              >
                Approve
              </Button>
              <Button
                danger
                size="small"
                icon={<XCircle className="w-4 h-4" />}
                onClick={handleReject}
                className="flex-1"
              >
                Reject
              </Button>
            </div>
          </div>
        )}

        {/* Timeline */}
        {timelineItems.length > 0 ? (
          <Timeline items={timelineItems} />
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              isIdle
                ? "Click Run to start the workflow"
                : "No steps executed yet"
            }
          />
        )}
      </div>

      {/* Run info */}
      {runId && (
        <div className="p-3 border-t border-border bg-surface">
          <div className="text-xs text-text-subtle">
            Run ID: <code className="font-mono">{runId}</code>
          </div>
        </div>
      )}
    </div>
  )
}

export default ExecutionPanel
