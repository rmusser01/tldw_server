import { Tabs } from "antd"

import { Badge, EmptyState } from "@/components/ui"
import type {
  WorkflowRunInvestigation,
  WorkflowStepAttempt
} from "@/types/workflow-editor"

interface WorkflowRunInspectorProps {
  investigation: WorkflowRunInvestigation | null
  className?: string
}

const formatAttemptSubtitle = (attempt: WorkflowStepAttempt) => {
  const details = [
    attempt.reason_code_core,
    attempt.error_summary,
    attempt.metadata?.retry_recommendation as string | undefined
  ].filter(Boolean)
  return details.join(" • ")
}

const getAttemptStatusVariant = (
  status: WorkflowStepAttempt["status"]
): "danger" | "success" | "warning" | "secondary" => {
  if (status === "failed") return "danger"
  if (status === "completed" || status === "success") return "success"
  if (status === "running") return "warning"
  return "secondary"
}

export const WorkflowRunInspector = ({
  investigation,
  className = ""
}: WorkflowRunInspectorProps) => {
  if (!investigation) {
    return (
      <EmptyState
        variant="inline"
        size="sm"
        title="No run diagnostics available"
      />
    )
  }

  const failure = investigation.primary_failure
  const attemptItems = investigation.attempts ?? []
  const recommendedActions = investigation.recommended_actions ?? []
  const evidenceJson = JSON.stringify(investigation.evidence ?? {}, null, 2)

  return (
    <section
      aria-label="Workflow run inspector"
      className={`rounded-lg border border-border bg-surface ${className}`}
    >
      <Tabs
        defaultActiveKey="summary"
        items={[
          {
            key: "summary",
            label: "Summary",
            children: (
              <div className="space-y-4 p-1">
                <div>
                  <h4 className="text-sm font-semibold text-text-muted">
                    Failure summary
                  </h4>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    {failure?.reason_code_core && (
                      <Badge variant="danger">{failure.reason_code_core}</Badge>
                    )}
                    {failure?.category && (
                      <Badge variant="secondary">{failure.category}</Badge>
                    )}
                    {failure?.blame_scope && (
                      <Badge variant="secondary">{failure.blame_scope}</Badge>
                    )}
                    {failure?.retryable !== undefined && (
                      <Badge
                        variant={failure.retryable ? "success" : "secondary"}
                      >
                        {failure.retryable ? "Retryable" : "Non-retryable"}
                      </Badge>
                    )}
                  </div>
                  {failure?.error_summary && (
                    <p className="mt-3 text-sm text-text-muted">
                      {failure.error_summary}
                    </p>
                  )}
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-text-muted">
                    Attempts
                  </h4>
                  {attemptItems.length > 0 ? (
                    <div className="mt-2 space-y-2">
                      {attemptItems.map((attempt) => (
                        <div
                          key={attempt.attempt_id}
                          className="rounded-md border border-border p-3"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-medium text-sm">
                              Attempt {attempt.attempt_number}
                            </span>
                            <Badge
                              variant={getAttemptStatusVariant(attempt.status)}
                            >
                              {attempt.status}
                            </Badge>
                          </div>
                          {formatAttemptSubtitle(attempt) && (
                            <p className="mt-2 text-xs text-text-subtle">
                              {formatAttemptSubtitle(attempt)}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-2 text-xs text-text-subtle">
                      No attempts recorded.
                    </p>
                  )}
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-text-muted">
                    Recommended actions
                  </h4>
                  {recommendedActions.length > 0 ? (
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-text-muted">
                      {recommendedActions.map((action) => (
                        <li key={action}>{action}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-xs text-text-subtle">
                      No recommended actions captured.
                    </p>
                  )}
                </div>
              </div>
            )
          },
          {
            key: "evidence",
            label: "Evidence",
            children: (
              <pre className="max-h-64 overflow-auto rounded-md bg-black/5 p-3 text-xs">
                {evidenceJson}
              </pre>
            )
          }
        ]}
      />
    </section>
  )
}

export default WorkflowRunInspector
