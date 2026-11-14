/**
 * ExecutionPanel Component
 *
 * Displays workflow execution status, step progress,
 * and handles human-in-the-loop approvals.
 */

import { useEffect, useMemo, useState } from "react"
import {
  Button,
  Progress,
  Timeline,
  Empty,
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
  Hand,
  ChevronRight
} from "lucide-react"
import { WorkflowRunInspector } from "@/components/Common/Workflow"
import { Alert } from "@/components/ui/primitives"
import type { StepExecutionStatus } from "@/types/workflow-editor"
import type { WorkflowEditorApi } from "@/hooks/useWorkflowEditorApi"
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
  api?: WorkflowEditorApi
}

export const ExecutionPanel = ({ className = "", api }: ExecutionPanelProps) => {
  const [inputJson, setInputJson] = useState(() => JSON.stringify(useWorkflowEditorStore.getState().pendingRunSubmission?.inputs ?? {}, null, 2))
  // Store state
  const nodes = useWorkflowEditorStore((s) => s.nodes)
  const runId = useWorkflowEditorStore((s) => s.runId)
  const pendingSubmission = useWorkflowEditorStore((s) => s.pendingRunSubmission)
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
  const resetExecution = useWorkflowEditorStore((s) => s.resetExecution)
  const loadRunInvestigation = useWorkflowEditorStore((s) => s.loadRunInvestigation)

  // Calculate progress
  const progress = useMemo(() => {
    const executable = nodes.filter((node) => node.data.stepType !== "start" && node.data.stepType !== "end")
    const total = executable.length
    if (total === 0) return 0

    const completed = executable.filter(
      (node) => nodeStates[node.id]?.status === "success" || nodeStates[node.id]?.status === "skipped"
    ).length

    return Math.round((completed / total) * 100)
  }, [nodes, nodeStates])

  const [now, setNow] = useState(Date.now)
  // Intentional nullish checks below: null/undefined mean absent; epoch 0 is valid.
  useEffect(() => {
    if (startedAt == null || completedAt != null) return
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [startedAt, completedAt])

  // Calculate elapsed time from the clock snapshot; completed runs stay fixed.
  const elapsed = useMemo(() => {
    if (startedAt == null) return null
    const end = completedAt ?? now
    const seconds = Math.floor((end - startedAt) / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`
  }, [startedAt, completedAt, now])

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
    void api?.run(inputJson)
  }

  const handleApprove = () => {
    if (!pendingApproval) return
    void api?.review("approve")
  }

  const handleReject = () => {
    if (!pendingApproval) return
    void api?.review("reject")
  }

  const isRunning = status === "running" || status === "queued"
  const isPaused = status === "paused"
  const isWaiting = status === "waiting_human" || status === "waiting_approval"
  const isIdle = status === "idle"
  const isCompleted = status === "completed" || status === "failed" || status === "cancelled"

  useEffect(() => {
    if (!api && status === "failed" && runId) void loadRunInvestigation(runId)
  }, [api, loadRunInvestigation, runId, status])

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
        {isIdle && <div className="mb-3"><label htmlFor="workflow-run-inputs" className="text-xs font-medium">Run inputs (JSON object)</label><TextArea id="workflow-run-inputs" aria-label="Run inputs" value={inputJson} onChange={(event) => setInputJson(event.target.value)} rows={4} className="font-mono" /></div>}
        {isIdle && pendingSubmission && <Alert variant="warning" title="Run submission has an unknown outcome" className="mb-3">Retry the original inputs to recover the same server run. Starting a new attempt may repeat work already accepted by the server.<Button size="small" disabled={api?.pending} onClick={() => { if (confirm("Forget this pending submission? A new Run may repeat work already accepted by the server.")) resetExecution() }}>Reset pending submission</Button></Alert>}
        <div className="flex items-center gap-2">
          {isIdle && (
            <Button
              type="primary"
              size="small"
              icon={<Play className="w-4 h-4" />}
              onClick={handleStartRun}
              disabled={!api?.ready || api.pending}
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
                onClick={() => void api?.control("pause")}
                disabled={!api?.ready || api.pending}
                className="flex-1"
              >
                Pause
              </Button>
              <Button
                danger
                size="small"
                icon={<Square className="w-4 h-4" />}
                onClick={() => void api?.control("cancel")}
                disabled={!api?.ready || api.pending}
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
                onClick={() => void api?.control("resume")}
                disabled={!api?.ready || api.pending}
                className="flex-1"
              >
                Resume
              </Button>
              <Button
                danger
                size="small"
                icon={<Square className="w-4 h-4" />}
                onClick={() => void api?.control("cancel")}
                disabled={!api?.ready || api.pending}
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
          {isWaiting && <Button danger size="small" disabled={!api?.ready || api.pending} onClick={() => void api?.control("cancel")}>Stop</Button>}
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
        {api?.error && <Alert variant="error" title="Workflow request failed" className="mb-3">{api.error}</Alert>}
        {runId && <p className="mb-3 text-xs break-all">Server run: {runId}</p>}
        {api?.output && <div className="mb-3"><h4 className="text-sm font-medium">Server output</h4><pre className="mt-2 whitespace-pre-wrap break-all rounded border border-border p-2 text-xs">{JSON.stringify(api.output, null, 2)}</pre></div>}
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
                  onClick: () => { if (api) void api.inspect(); else void loadRunInvestigation(runId) }
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
                disabled={!api?.ready || api.pending}
                className="flex-1"
              >
                Approve
              </Button>
              <Button
                danger
                size="small"
                icon={<XCircle className="w-4 h-4" />}
                onClick={handleReject}
                disabled={!api?.ready || api.pending}
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
