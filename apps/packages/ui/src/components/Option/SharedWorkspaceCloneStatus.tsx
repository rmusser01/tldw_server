import React from "react"
import { Button } from "antd"
import { ExportOutlined, ReloadOutlined } from "@ant-design/icons"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import type { CloneRowState } from "@/services/shared-clone-manager"

const phaseCopy = {
  queued: "Waiting to copy...",
  authorizing: "Checking access...",
  preparing: "Preparing the copy...",
  sources: "Copying sources...",
  notes: "Copying notes...",
  artifacts: "Copying saved outputs...",
  finalizing: "Finishing the copy..."
}
const warningCopy: Record<string, string> = {
  source_copy_failed: "Sources could not be copied ({{count}}).",
  note_copy_failed: "Notes could not be copied ({{count}}).",
  artifact_copy_failed: "Saved outputs could not be copied ({{count}}).",
  media_copy_failed: "Media could not be copied ({{count}}).",
  membership_skipped: "Unsupported workspace items were skipped ({{count}}).",
  vector_index_not_generated: "Vector indexes were not copied ({{count}})."
}
const failureCopy: Record<string, string> = {
  clone_access_revoked: "Access ended before the copy completed.",
  clone_permission_removed: "The owner disabled copying before it completed.",
  clone_cancelled: "The copy was cancelled.",
  clone_interrupted: "The copy was interrupted.",
  lease_expired: "The copy was interrupted.",
  clone_persistence_failed: "The copy could not be saved.",
  source_snapshot_unavailable: "The source snapshot was unavailable."
}

export function SharedWorkspaceCloneStatus({
  row,
  label,
  canRetry,
  onRetry,
  onRefresh
}: {
  row: CloneRowState
  label: string
  canRetry: boolean
  onRetry: () => void
  onRefresh: () => void
}) {
  const { t } = useTranslation("common")
  const navigate = useNavigate()
  const { operation, issue } = row
  const result = operation?.status === "succeeded" ? operation.result : null
  const failure = operation?.status === "failed" ? operation.error : null
  const progress = operation?.progress
  const active = !result && !failure
  const knownFailure =
    failure && Object.prototype.hasOwnProperty.call(failureCopy, failure.code)

  return (
    <div className="mt-3 min-w-0 space-y-2 text-sm [overflow-wrap:anywhere]">
      <div role="status" aria-live="polite" aria-atomic="true">
        {result ? (
          <p className="m-0 font-medium">
            {result.outcome === "complete"
              ? t("sharedWithMe.clone.complete", "Copy ready.")
              : t("sharedWithMe.clone.partial", "Copy ready with omissions.")}
          </p>
        ) : failure ? (
          <p className="m-0 font-medium">
            {knownFailure
              ? t(
                  `sharedWithMe.clone.errors.${failure.code}`,
                  failureCopy[failure.code]
                )
              : t(
                  "sharedWithMe.clone.failed",
                  "The copy could not be completed."
                )}
          </p>
        ) : issue ? (
          <p className="m-0">
            {issue === "uncertain"
              ? t(
                  "sharedWithMe.clone.uncertain",
                  "Copy status could not be confirmed. Checking again will recover the same operation."
                )
              : issue === "recovery_full"
                ? t(
                    "sharedWithMe.clone.recoveryFull",
                    "The recovery list is full. No new copy was started; try again after older records expire."
                  )
                : issue === "rejected"
                  ? t(
                      "sharedWithMe.clone.rejected",
                      "The copy request was not accepted. Refresh your access before trying again."
                    )
                  : t(
                      "sharedWithMe.clone.unavailable",
                      "This copy's status is unavailable. No new copy has been started."
                    )}
          </p>
        ) : (
          <p className="m-0">
            {progress
              ? t(
                  `sharedWithMe.clone.phases.${progress.phase}`,
                  phaseCopy[progress.phase]
                )
              : t(
                  "sharedWithMe.clone.submitting",
                  "Confirming the copy request..."
                )}
          </p>
        )}
      </div>
      {active && progress && !issue && (
        <progress
          className="block h-2 w-full max-w-sm"
          value={progress.percent}
          max={100}
          aria-label={t("sharedWithMe.clone.progressLabel", {
            defaultValue: "Copy progress for {{name}}",
            name: label
          })}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress.percent}
        />
      )}
      {result && (
        <>
          <p className="m-0 text-text-muted">
            {t("sharedWithMe.clone.counts", {
              defaultValue:
                "{{sources}} of {{totalSources}} sources copied; {{notes}} of {{totalNotes}} notes; {{artifacts}} of {{totalArtifacts}} saved outputs.",
              sources: result.counts.sources_copied,
              totalSources: result.counts.sources_attempted,
              notes: result.counts.notes_copied,
              totalNotes: result.counts.notes_attempted,
              artifacts: result.counts.artifacts_copied,
              totalArtifacts: result.counts.artifacts_attempted
            })}
          </p>
          <p className="m-0 text-text-muted">
            {t(
              `sharedWithMe.clone.textSearch.${result.readiness.text_search}`,
              result.readiness.text_search === "ready"
                ? "Text search ready."
                : "Text search unavailable."
            )}{" "}
            {t(
              `sharedWithMe.clone.citations.${result.readiness.citations}`,
              result.readiness.citations === "ready"
                ? "Citations ready."
                : "Citations unavailable."
            )}
          </p>
          <p className="m-0 text-text-muted">
            {t(
              `sharedWithMe.clone.vectorSearch.${result.readiness.vector_search}`,
              result.readiness.vector_search === "ready"
                ? "Vector search ready."
                : result.readiness.vector_search === "needs_indexing"
                  ? "Vector search needs indexing."
                  : "Vector search is not configured."
            )}
          </p>
          {result.warnings.length > 0 && (
            <ul className="m-0 list-disc pl-5 text-text-muted">
              {result.warnings.map((warning, index) => {
                const known = Object.prototype.hasOwnProperty.call(
                  warningCopy,
                  warning.code
                )
                return (
                  <li key={`${warning.code}-${index}`}>
                    {t(
                      known
                        ? `sharedWithMe.clone.warnings.${warning.code}`
                        : "sharedWithMe.clone.warning",
                      {
                        defaultValue: known
                          ? warningCopy[warning.code]
                          : "Some items need attention ({{count}}).",
                        count: warning.count
                      }
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </>
      )}
      {failure && failure.cleanup_state !== "complete" && (
        <p className="m-0 text-text-muted">
          {failure.cleanup_state === "pending"
            ? t(
                "sharedWithMe.clone.cleanupPending",
                "Temporary copy data is being cleaned up."
              )
            : t(
                "sharedWithMe.clone.cleanupUnknown",
                "Cleanup could not be confirmed. Contact your server administrator."
              )}
        </p>
      )}
      {!row.recoveryAvailable && (
        <p className="m-0 text-text-muted">
          {t(
            "sharedWithMe.clone.storageUnavailable",
            "Reload recovery is unavailable in this browser. Keep this page open until the copy finishes."
          )}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {result && (
          <Button
            icon={<ExportOutlined />}
            onClick={() =>
              navigate(
                `/research-workspace?workspace=${encodeURIComponent(result.workspace_id)}`
              )
            }
            aria-label={t("sharedWithMe.clone.openLabel", {
              defaultValue: "Open copy of {{name}}",
              name: label
            })}
          >
            {t("sharedWithMe.clone.open", "Open copy")}
          </Button>
        )}
        {failure && operation?.retryable && canRetry && (
          <Button
            icon={<ReloadOutlined />}
            onClick={onRetry}
            aria-label={t("sharedWithMe.clone.retryLabel", {
              defaultValue: "Retry copy of {{name}}",
              name: label
            })}
          >
            {t("sharedWithMe.clone.retry", "Retry copy")}
          </Button>
        )}
        {(issue === "uncertain" ||
          issue === "unavailable" ||
          (failure && failure.cleanup_state !== "complete")) && (
          <Button
            icon={<ReloadOutlined />}
            onClick={onRefresh}
            loading={row.pending}
            aria-label={t("sharedWithMe.clone.checkLabel", {
              defaultValue: "Check copy status for {{name}}",
              name: label
            })}
          >
            {t("sharedWithMe.clone.check", "Check status")}
          </Button>
        )}
      </div>
    </div>
  )
}
