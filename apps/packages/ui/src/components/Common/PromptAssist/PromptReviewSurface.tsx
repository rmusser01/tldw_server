import { Button } from "@/components/Common/Button"
import type {
  PromptImproveFinding,
  PromptResolvedModel
} from "@/services/prompt-improvement"
import { useEffect, useId, useState } from "react"
import { useTranslation } from "react-i18next"

import { PromptDiff } from "./PromptDiff"
import type { PromptAssistNotice } from "./prompt-assist-state"

export type PromptReviewSurfaceProps = {
  original: string
  candidate: string
  findings: readonly PromptImproveFinding[]
  warnings: readonly string[]
  notice: PromptAssistNotice
  resolvedModel: PromptResolvedModel
  replaceConfirmationRequired?: boolean
  mode?: "review" | "inspection"
  onCandidateChange: (candidate: string) => void
  onApply: () => void
  onConfirmReplace: () => void
  onCancel: () => void
  onUndo?: () => void
  onEscape?: () => void
}

const knownWarningCodes = new Set([
  "unstructured_output",
  "target_mismatch",
  "placeholder_mismatch",
  "url_mismatch",
  "protected_token_mismatch",
  "code_fence_mismatch",
  "wrapper_mismatch",
  "large_rewrite"
])

const noticeCopy = (notice: PromptAssistNotice): string | null => {
  switch (notice) {
    case "draft_changed":
      return "The draft changed while this result was open. Applying normally will not overwrite it."
    case "route_changed":
      return "The active model changed while this result was open. Review before applying."
    case "review_required":
      return "Review the safety notices before applying."
    default:
      return null
  }
}

export function PromptReviewSurface({
  original,
  candidate,
  findings,
  warnings,
  notice,
  resolvedModel,
  replaceConfirmationRequired = false,
  mode = "review",
  onCandidateChange,
  onApply,
  onConfirmReplace,
  onCancel,
  onUndo,
  onEscape
}: PromptReviewSurfaceProps) {
  const { t } = useTranslation(["common"])
  const [activeTab, setActiveTab] = useState<"edit" | "changes">("edit")
  const [confirmingReplace, setConfirmingReplace] = useState(false)
  const [copyStatus, setCopyStatus] = useState("")
  const [editedCandidate, setEditedCandidate] = useState(candidate)
  const observationsId = useId()
  const viewPanelId = useId()
  const editButtonId = useId()
  const changesButtonId = useId()
  const isInspection = mode === "inspection"
  const renderedNotice = noticeCopy(notice)
  const hasSafetyNotice = warnings.length > 0 || notice === "review_required"

  useEffect(() => {
    setEditedCandidate(candidate)
  }, [candidate])

  const copyCandidate = async () => {
    try {
      await navigator.clipboard.writeText(editedCandidate)
      setCopyStatus(t("common:promptAssist.copied", "Candidate copied."))
    } catch {
      setCopyStatus(
        t("common:promptAssist.copyFailed", "Could not copy the candidate.")
      )
    }
  }

  return (
    <div
      className="space-y-5"
      onKeyDown={(event) => {
        if (event.key !== "Escape") return
        event.preventDefault()
        event.stopPropagation()
        if (confirmingReplace) {
          setConfirmingReplace(false)
        } else {
          onEscape?.()
        }
      }}>
      <div className="space-y-1">
        <h2 className="text-base font-semibold">
          {isInspection
            ? t("common:promptAssist.appliedChanges", "Applied changes")
            : t("common:promptAssist.reviewTitle", "Review improved prompt")}
        </h2>
        <p className="text-sm text-muted-foreground">
          {t("common:promptAssist.usedModel", "Used {{model}} ({{provider}})", {
            model: resolvedModel.display_name,
            provider: resolvedModel.provider
          })}
        </p>
      </div>

      {renderedNotice ? (
        <div
          role={notice === "draft_changed" ? "alert" : "status"}
          className="rounded-md border border-warn/30 bg-warn/10 p-3 text-sm text-text">
          {t(`common:promptAssist.notice.${notice}`, renderedNotice)}
        </div>
      ) : null}
      {hasSafetyNotice && notice !== "review_required" ? (
        <p className="text-sm font-medium text-warn">
          {t(
            "common:promptAssist.reviewSafety",
            "Review the safety notices before applying."
          )}
        </p>
      ) : null}
      {warnings.length > 0 ? (
        <ul
          className="space-y-1 text-sm text-muted-foreground"
          aria-label={t("common:promptAssist.safetyNotices", "Safety notices")}>
          {warnings.slice(0, 5).map((warning, index) => (
            <li key={`${warning}-${index}`}>
              •{" "}
              {t(
                knownWarningCodes.has(warning)
                  ? `common:promptAssist.warnings.${warning}`
                  : "common:promptAssist.warnings.unknown",
                "Prompt safety notice."
              )}
            </li>
          ))}
        </ul>
      ) : null}

      {findings.length > 0 ? (
        <section aria-labelledby={observationsId}>
          <h3 id={observationsId} className="mb-2 text-sm font-semibold">
            {t("common:promptAssist.observations", "Model observations")}
          </h3>
          <ul
            aria-label={t(
              "common:promptAssist.observations",
              "Model observations"
            )}
            className="space-y-2">
            {findings.slice(0, 5).map((finding, index) => (
              <li
                key={`${finding.category}-${index}`}
                className="rounded-md bg-muted/50 p-3 text-sm">
                <span className="font-medium">{finding.issue}</span>
                <span className="block text-muted-foreground">
                  {finding.change}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <div>
        <div
          role="group"
          aria-label={t("common:promptAssist.reviewViews", "Review views")}
          className="mb-3 flex gap-1 border-b border-border">
          <button
            id={editButtonId}
            type="button"
            aria-pressed={activeTab === "edit"}
            aria-controls={viewPanelId}
            className="min-h-11 border-b-2 border-transparent px-4 text-sm font-medium transition-colors motion-reduce:transition-none aria-pressed:border-text aria-pressed:text-text aria-pressed:font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            onClick={() => setActiveTab("edit")}>
            {t("common:promptAssist.editTab", "Edit")}
          </button>
          <button
            id={changesButtonId}
            type="button"
            aria-pressed={activeTab === "changes"}
            aria-controls={viewPanelId}
            className="min-h-11 border-b-2 border-transparent px-4 text-sm font-medium transition-colors motion-reduce:transition-none aria-pressed:border-text aria-pressed:text-text aria-pressed:font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            onClick={() => setActiveTab("changes")}>
            {t("common:promptAssist.changesTab", "Changes")}
          </button>
        </div>
        <div
          id={viewPanelId}
          aria-labelledby={
            activeTab === "edit" ? editButtonId : changesButtonId
          }>
          {activeTab === "edit" ? (
            <textarea
              aria-label={t(
                "common:promptAssist.candidateLabel",
                "Improved prompt candidate"
              )}
              className="min-h-56 w-full resize-y rounded-md border border-border bg-bg p-3 text-sm leading-6"
              readOnly={isInspection}
              value={editedCandidate}
              onChange={(event) => {
                setEditedCandidate(event.target.value)
                onCandidateChange(event.target.value)
              }}
            />
          ) : (
            <PromptDiff original={original} candidate={editedCandidate} />
          )}
        </div>
      </div>

      {copyStatus ? (
        <p
          role="status"
          aria-live="polite"
          aria-atomic="true"
          className="text-sm">
          {copyStatus}
        </p>
      ) : null}

      {confirmingReplace ? (
        <div className="rounded-md border border-warn/30 bg-warn/10 p-3">
          <p className="mb-3 text-sm font-medium">
            {t(
              "common:promptAssist.confirmReplaceQuestion",
              "Replace the current draft with this candidate?"
            )}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="danger"
              size="lg"
              disabled={!editedCandidate.trim()}
              onClick={onConfirmReplace}>
              {t("common:promptAssist.confirmReplace", "Confirm replace")}
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => setConfirmingReplace(false)}>
              {t("common:cancel", "Cancel")}
            </Button>
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap justify-end gap-2">
        <Button variant="outline" size="lg" onClick={copyCandidate}>
          {t("common:promptAssist.copy", "Copy")}
        </Button>
        {isInspection ? (
          <>
            {onUndo ? (
              <Button variant="outline" size="lg" onClick={onUndo}>
                {t("common:promptAssist.undo", "Undo improvement")}
              </Button>
            ) : null}
            <Button
              variant="primary"
              size="lg"
              className="!bg-text !text-bg hover:!bg-text active:!bg-text"
              onClick={onCancel}>
              {t("common:close", "Close")}
            </Button>
          </>
        ) : (
          <>
            <Button variant="outline" size="lg" onClick={onCancel}>
              {t("common:cancel", "Cancel")}
            </Button>
            {replaceConfirmationRequired ? (
              <Button
                variant="outline"
                size="lg"
                className="border-danger text-danger"
                disabled={!editedCandidate.trim()}
                onClick={() => setConfirmingReplace(true)}>
                {t(
                  "common:promptAssist.replaceCurrent",
                  "Replace current draft"
                )}
              </Button>
            ) : (
              <Button
                variant="primary"
                size="lg"
                className="!bg-text !text-bg hover:!bg-text active:!bg-text"
                disabled={!editedCandidate.trim()}
                onClick={onApply}>
                {t("common:promptAssist.apply", "Apply to draft")}
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  )
}
