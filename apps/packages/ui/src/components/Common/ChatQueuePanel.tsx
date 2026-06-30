import React from "react"
import type { TFunction } from "i18next"
import { useTranslation } from "react-i18next"

import { getDesignSystemState } from "@/design-system"
import type { QueuedRequest } from "@/utils/chat-request-queue"

type ChatQueuePanelProps = {
  queue: QueuedRequest[]
  isConnectionReady: boolean
  isStreaming: boolean
  onRunNext: () => void | Promise<void>
  onRunNow: (requestId: string) => void | Promise<void>
  onDelete: (requestId: string) => void
  onMove: (requestId: string, direction: "up" | "down") => void
  onUpdate: (requestId: string, promptText: string) => void
  onClearAll: () => void
  onOpenDiagnostics?: () => void
  forceRunDisabledReason?: string | null
}

const BLOCKED_STATE_LABEL = getDesignSystemState("blocked").label

const formatBlockedReason = (
  blockedReason: string | null,
  t: TFunction
) => {
  switch (blockedReason) {
    case "unsupported_attachment":
      return t(
        "playground:composer.queue.unsupportedAttachment",
        "Needs attachment repair in the composer"
      )
    case "dispatch_failed":
      return t(
        "playground:composer.queue.dispatchFailed",
        "Dispatch failed. Review and retry."
      )
    default:
      return blockedReason
        ? blockedReason
        : t("playground:composer.queue.blocked", BLOCKED_STATE_LABEL)
  }
}

const summarizeQueueItem = (item: QueuedRequest) => {
  const trimmed = item.promptText.trim()
  if (!trimmed) return "Untitled request"
  return trimmed.length > 72 ? `${trimmed.slice(0, 69)}...` : trimmed
}

export const ChatQueuePanel: React.FC<ChatQueuePanelProps> = ({
  queue,
  isConnectionReady,
  isStreaming,
  onRunNext,
  onRunNow,
  onDelete,
  onMove,
  onUpdate,
  onClearAll,
  onOpenDiagnostics,
  forceRunDisabledReason
}) => {
  const { t } = useTranslation(["playground", "settings", "common"])
  const [expanded, setExpanded] = React.useState(false)
  const [editingRequestId, setEditingRequestId] = React.useState<string | null>(
    null
  )
  const [draftPrompt, setDraftPrompt] = React.useState("")

  if (queue.length === 0) {
    return null
  }

  const nextItem = queue[0]
  const runNextDisabled =
    !isConnectionReady ||
    nextItem.status === "sending" ||
    (isStreaming && Boolean(forceRunDisabledReason))

  return (
    <div className="mt-2 space-y-2 rounded-2xl border border-success/25 bg-success/10 px-3 py-3 text-xs text-success">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="font-semibold">
            {t("playground:composer.queue.summary", "{{count}} queued", {
              count: queue.length
            })}
          </div>
          <div className="mt-1 text-success/90">
            <span className="font-medium">
              {t("playground:composer.queue.next", "Next:")}
            </span>{" "}
            {summarizeQueueItem(nextItem)}
          </div>
          {nextItem.status === "blocked" && (
            <div className="mt-1 text-warning">
              {formatBlockedReason(nextItem.blockedReason, t)}
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setExpanded((value) => !value)}
            className="rounded-md border border-success/30 bg-surface px-2 py-1 font-medium text-success hover:bg-success/10"
          >
            {expanded
              ? t("common:hide", "Hide queue")
              : t("playground:composer.queue.view", "View queue")}
          </button>
          <button
            type="button"
            onClick={() => void onRunNext()}
            disabled={runNextDisabled}
            title={
              isStreaming && forceRunDisabledReason
                ? forceRunDisabledReason
                : undefined
            }
            className={`rounded-md border border-success/30 bg-surface px-2 py-1 font-medium text-success hover:bg-success/10 ${
              runNextDisabled
                ? "cursor-not-allowed opacity-60"
                : ""
            }`}
          >
            {nextItem.status === "blocked"
              ? t("playground:composer.queue.retryNext", "Retry next")
              : t("playground:composer.queue.runNext", "Run next")}
          </button>
          {onOpenDiagnostics && (
            <button
              type="button"
              onClick={onOpenDiagnostics}
              className="font-medium underline hover:text-success"
            >
              {t("settings:healthSummary.diagnostics", "Health & diagnostics")}
            </button>
          )}
          <button
            type="button"
            onClick={onClearAll}
            className="font-medium underline hover:text-success"
          >
            {t("playground:composer.queue.clearAll", "Clear all")}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="space-y-2 border-t border-success/20 pt-2">
          {queue.map((item, index) => {
            const isEditing = editingRequestId === item.id
            const isSendingItem = item.status === "sending"
            const runNowDisabled =
              isSendingItem ||
              (isStreaming && Boolean(forceRunDisabledReason))
            const runNowTitle =
              isStreaming && forceRunDisabledReason
                ? forceRunDisabledReason
                : undefined
            const moveUpDisabled = isSendingItem || index === 0
            const moveDownDisabled = isSendingItem || index === queue.length - 1

            return (
              <div
                key={item.id}
                className="rounded-xl border border-success/20 bg-surface/80 px-3 py-3 text-text"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0 flex-1 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-full bg-surface2 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-text-subtle">
                        {item.status}
                      </span>
                      {item.snapshot.selectedModel && (
                        <span className="rounded-full border border-border px-2 py-0.5 text-[11px] text-text-subtle">
                          {item.snapshot.selectedModel}
                        </span>
                      )}
                      <span className="rounded-full border border-border px-2 py-0.5 text-[11px] text-text-subtle">
                        {item.snapshot.chatMode}
                      </span>
                      {item.attachments.length > 0 && (
                        <span className="rounded-full border border-border px-2 py-0.5 text-[11px] text-text-subtle">
                          {t(
                            "playground:composer.queue.attachments",
                            "{{count}} attachment(s)",
                            { count: item.attachments.length }
                          )}
                        </span>
                      )}
                    </div>
                    {isEditing ? (
                      <label className="block">
                        <span className="sr-only">
                          {t(
                            "playground:composer.queue.editLabel",
                            "Edit queued request"
                          )}
                        </span>
                        <textarea
                          aria-label={t(
                            "playground:composer.queue.editLabel",
                            "Edit queued request"
                          )}
                          value={draftPrompt}
                          onChange={(event) => setDraftPrompt(event.target.value)}
                          className="min-h-[5rem] w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text"
                        />
                      </label>
                    ) : (
                      <p className="whitespace-pre-wrap break-words text-sm">
                        {item.promptText}
                      </p>
                    )}
                    {item.status === "blocked" && (
                      <p className="text-[11px] text-warning">
                        {formatBlockedReason(item.blockedReason, t)}
                      </p>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    {isEditing ? (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            onUpdate(item.id, draftPrompt)
                            setEditingRequestId(null)
                            setDraftPrompt("")
                          }}
                          className="rounded-md border border-primary/30 bg-primary/10 px-2 py-1 font-medium text-primaryStrong hover:bg-primary/15"
                        >
                          {t("common:save", "Save")}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setEditingRequestId(null)
                            setDraftPrompt("")
                          }}
                          className="rounded-md border border-border px-2 py-1 font-medium text-text-subtle hover:bg-surface2"
                        >
                          {t("common:cancel", "Cancel")}
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            setEditingRequestId(item.id)
                            setDraftPrompt(item.promptText)
                          }}
                          disabled={isSendingItem}
                          className={`rounded-md border border-border px-2 py-1 font-medium text-text-subtle hover:bg-surface2 ${
                            isSendingItem ? "cursor-not-allowed opacity-50" : ""
                          }`}
                        >
                          {t("common:edit", "Edit")}
                        </button>
                        <button
                          type="button"
                          onClick={() => onMove(item.id, "up")}
                          disabled={moveUpDisabled}
                          className={`rounded-md border border-border px-2 py-1 font-medium text-text-subtle hover:bg-surface2 ${
                            moveUpDisabled ? "cursor-not-allowed opacity-50" : ""
                          }`}
                        >
                          {t("playground:composer.queue.moveUp", "Move up")}
                        </button>
                        <button
                          type="button"
                          onClick={() => onMove(item.id, "down")}
                          disabled={moveDownDisabled}
                          className={`rounded-md border border-border px-2 py-1 font-medium text-text-subtle hover:bg-surface2 ${
                            moveDownDisabled ? "cursor-not-allowed opacity-50" : ""
                          }`}
                        >
                          {t("playground:composer.queue.moveDown", "Move down")}
                        </button>
                        <button
                          type="button"
                          onClick={() => void onRunNow(item.id)}
                          disabled={runNowDisabled}
                          title={runNowTitle}
                          className={`rounded-md border border-success/30 bg-success/10 px-2 py-1 font-medium text-success hover:bg-success/15 ${
                            runNowDisabled
                              ? "cursor-not-allowed opacity-50"
                              : ""
                          }`}
                        >
                          {isStreaming
                            ? t(
                                "playground:composer.queue.cancelAndRunNow",
                                "Cancel current & run now"
                              )
                            : item.status === "blocked"
                              ? t("playground:composer.queue.retryNow", "Retry now")
                              : t("playground:composer.queue.runNow", "Run now")}
                        </button>
                        <button
                          type="button"
                          onClick={() => onDelete(item.id)}
                          disabled={item.status === "sending"}
                          className={`rounded-md border border-border px-2 py-1 font-medium text-text-subtle hover:bg-surface2 ${
                            item.status === "sending"
                              ? "cursor-not-allowed opacity-50"
                              : ""
                          }`}
                        >
                          {t("common:delete", "Delete")}
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default ChatQueuePanel
