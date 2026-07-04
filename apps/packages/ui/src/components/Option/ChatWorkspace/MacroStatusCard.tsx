import React from "react"

import type {
  ChatMacroBranchSummary,
  ChatMacroRunDetailResponse
} from "@/services/chat-macros"

export interface ChatMacroStatusMetadata {
  run_id: string
  name?: string | null
  command?: string | null
  status: string
  detail_url?: string | null
  output_profile?: string | null
  branch_count?: number | null
}

export interface MacroStatusCardProps {
  metadata: ChatMacroStatusMetadata
  runDetail?: ChatMacroRunDetailResponse | null
  onCancel?: (runId: string) => void
  onOpenDetail?: (runId: string) => void
  className?: string
}

const CANCELLABLE_STATUSES = new Set(["pending", "queued", "running", "processing"])

const redactSensitiveText = (value: string): string =>
  value
    .replace(/Authorization:\s*Bearer\s+\S+/gi, "[redacted bearer token]")
    .replace(/\b(?:api[_-]?key|x-api-key|token)\s*[:=]\s*["']?[^"',\s}]+/gi, "[redacted secret]")
    .replace(/\bsk-[A-Za-z0-9_-]{8,}\b/g, "[redacted secret]")
    .replace(/\bAIza[0-9A-Za-z_-]{8,}\b/g, "[redacted secret]")

const branchLabel = (branch: ChatMacroBranchSummary): string =>
  branch.label || branch.output_name || branch.step_id || branch.branch_id

const formatBranchCount = (count: number): string =>
  count === 1 ? "1 branch" : `${count} branches`

export const MacroStatusCard = ({
  metadata,
  runDetail,
  onCancel,
  onOpenDetail,
  className
}: MacroStatusCardProps) => {
  const run = runDetail?.run
  const command = metadata.command || metadata.name || run?.macro_command || run?.macro_name || "macro"
  const status = run?.status || metadata.status
  const outputProfile = run?.output_profile || metadata.output_profile || "default"
  const branches = runDetail?.branches ?? []
  const branchCount =
    branches.length > 0
      ? branches.length
      : typeof metadata.branch_count === "number"
        ? metadata.branch_count
        : null
  const canCancel = Boolean(onCancel && CANCELLABLE_STATUSES.has(status))
  const canOpenDetail = Boolean(onOpenDetail)

  return (
    <article
      className={[
        "rounded-md border border-border bg-surface px-3 py-3 text-sm text-text shadow-sm",
        className
      ]
        .filter(Boolean)
        .join(" ")}
      aria-label={`/${command} macro run ${status}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold">/{command}</span>
            <span className="rounded-sm border border-border bg-background px-2 py-0.5 text-xs font-medium text-text-muted">
              {status}
            </span>
          </div>
          <p className="mt-1 break-all text-xs text-text-muted">
            Run {metadata.run_id}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {canOpenDetail ? (
            <button
              type="button"
              className="inline-flex min-h-[32px] items-center rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => onOpenDetail?.(metadata.run_id)}
              aria-label="View macro run detail"
            >
              Details
            </button>
          ) : null}
          {canCancel ? (
            <button
              type="button"
              className="inline-flex min-h-[32px] items-center rounded-md border border-danger/50 px-3 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => onCancel?.(metadata.run_id)}
              aria-label="Cancel macro run"
            >
              Cancel
            </button>
          ) : null}
        </div>
      </div>

      <dl className="mt-3 grid gap-2 text-xs text-text-muted sm:grid-cols-2">
        {branchCount !== null ? (
          <div>
            <dt className="font-medium text-text-muted">Branches</dt>
            <dd className="text-text">{formatBranchCount(branchCount)}</dd>
          </div>
        ) : null}
        <div>
          <dt className="font-medium text-text-muted">Output profile</dt>
          <dd className="text-text">{outputProfile}</dd>
        </div>
      </dl>

      {branches.length > 0 ? (
        <ul className="mt-3 space-y-2">
          {branches.map((branch) => {
            const safeError = branch.error ? redactSensitiveText(branch.error) : null
            return (
              <li
                key={branch.branch_id}
                className="rounded-md border border-border bg-background px-3 py-2"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{branchLabel(branch)}</span>
                  <span className="text-xs text-text-muted">
                    Branch status: {branch.status}
                  </span>
                  {branch.error_code ? (
                    <span className="rounded-sm bg-danger/10 px-2 py-0.5 text-xs font-medium text-danger">
                      {branch.error_code}
                    </span>
                  ) : null}
                </div>
                {safeError ? (
                  <p className="mt-1 break-words text-xs text-text-muted">
                    {safeError}
                  </p>
                ) : null}
              </li>
            )
          })}
        </ul>
      ) : null}
    </article>
  )
}

export const isChatMacroStatusComplete = (status: string | null | undefined): boolean =>
  status === "completed" || status === "posted"
